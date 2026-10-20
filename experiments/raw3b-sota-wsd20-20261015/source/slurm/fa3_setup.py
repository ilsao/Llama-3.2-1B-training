"""Compute-only reproducible FA3 build; no GPU/model checks charged as setup."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_state import atomic_json, file_digest

SHA = '94e22c906678e5483fa0e9e24d8e787bc2c0ed4c'


def run(command, **kwargs):
    subprocess.run([str(x) for x in command], check=True, **kwargs)


def main():
    from train import require_compute
    require_compute()
    root = Path(os.environ['DATA_ROOT'])
    prefix = Path(os.environ['TRAIN_ENV_PREFIX'])
    expected_prefix = root / 'environments/training-muon-torch291-fa3-94e22c9'
    if prefix.resolve() != expected_prefix.resolve():
        raise ValueError('Native Muon setup must use its separate pinned environment')
    if (prefix / 'fa3-build.json').exists():
        if (json.loads((prefix / 'fa3-build.json').read_text())['source_commit'] != SHA or
                json.loads((prefix / 'fa3-build.json').read_text())['versions']['torch'].split('+')[0] != '2.9.1'):
            raise ValueError('Existing FA3 environment has another identity')
        return
    if not (prefix / 'bin/python').exists():
        run(['/work/envstack/apps/miniconda3/26.1.1/condabin/conda', 'create', '--yes',
             '--clone', root / 'environments/training-py311', '--prefix', prefix])
    python = prefix / 'bin/python'
    run([python, '-m', 'pip', 'install', '--cache-dir', root / 'caches/pip',
         'torch==2.9.1', '--index-url', 'https://download.pytorch.org/whl/cu126'])
    run([python, '-m', 'pip', 'install', '--cache-dir', root / 'caches/pip',
         '-r', Path(__file__).resolve().parents[1] / 'requirements-training.txt'])
    run([python, '-m', 'pip', 'install', '--cache-dir', root / 'caches/pip',
         '-r', Path(__file__).resolve().parents[1] / 'requirements-fa3-build.txt'])
    source = root / 'caches/fa3-source/torch291' / SHA
    source.parent.mkdir(parents=True, exist_ok=True)
    if not (source / '.git').exists():
        source.mkdir(exist_ok=True)
        run(['git', 'init', source])
        run(['git', '-C', source, 'remote', 'add', 'origin', 'https://github.com/Dao-AILab/flash-attention.git'])
    run(['git', '-C', source, 'fetch', '--depth=1', 'origin', SHA])
    run(['git', '-C', source, 'checkout', '--detach', SHA])
    run(['git', '-C', source, 'submodule', 'update', '--init', 'csrc/cutlass'])
    wheels = root / 'caches/fa3-wheels/torch291-cu126' / SHA
    wheels.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MAX_JOBS='8', NVCC_THREADS='1', FLASH_ATTENTION_FORCE_BUILD='TRUE')
    env['PATH'] = str(prefix / 'bin') + os.pathsep + env['PATH']
    cache = root / 'caches/flash-attention'
    cache.mkdir(parents=True, exist_ok=True)
    # Cover the upstream cache setting and XDG defaults without changing HOME.
    env['FLASH_ATTENTION_HOME'] = str(cache)
    for feature in ('FP16', 'FP8', 'SM80', 'HDIM96', 'HDIM128', 'HDIM192', 'HDIM256',
                    'HDIMDIFF64', 'HDIMDIFF192', 'PAGEDKV', 'APPENDKV', 'LOCAL', 'SOFTCAP'):
        env['FLASH_ATTENTION_DISABLE_' + feature] = 'TRUE'
    run([python, '-m', 'pip', 'wheel', '--no-build-isolation', '--no-deps',
         '--wheel-dir', wheels, str(source / 'hopper')], env=env)
    built = list(wheels.glob('*.whl'))
    if len(built) != 1:
        raise ValueError('Expected exactly one FA3 wheel')
    run([python, '-m', 'pip', 'install', '--no-deps', built[0]], env=env)
    run([python, '-m', 'pip', 'check'], env=env)
    freeze = subprocess.check_output([python, '-m', 'pip', 'freeze', '--all'], text=True, env=env)
    (prefix / 'requirements-resolved.txt').write_text(freeze)
    metadata = json.loads(subprocess.check_output([python, '-c',
        'import importlib.metadata,json,torch; print(json.dumps({"versions":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},"torch_cuda":torch.version.cuda,"torch_abi":torch._C._GLIBCXX_USE_CXX11_ABI,"native_muon":hasattr(torch.optim,"Muon")}))'], text=True, env=env))
    if metadata['versions']['torch'].split('+')[0] != '2.9.1' or metadata['versions']['transformers'] != '4.51.3':
        raise ValueError('Pinned training framework changed during installation')
    if not metadata['native_muon']:
        raise ValueError('Installed PyTorch does not expose native Muon')
    metadata.update(source_commit=SHA, wheel=str(built[0]), wheel_sha256=file_digest(built[0]),
                    submodules=subprocess.check_output(['git', '-C', source, 'submodule', 'status'], text=True),
                    driver=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True),
                    nvcc=subprocess.check_output(['/work/envstack/apps/cuda/12.6/bin/nvcc', '--version'], text=True),
                    build_environment={k:v for k,v in env.items() if k.startswith('FLASH_ATTENTION_') or k in ('MAX_JOBS','NVCC_THREADS')})
    atomic_json(prefix / 'fa3-build.json', metadata)


if __name__ == '__main__':
    main()
