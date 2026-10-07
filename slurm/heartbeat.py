#!/usr/bin/env python3
"""Project-scoped 20-minute Codex continuity; only lightweight orchestration here."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import shutil
import subprocess
import sys
import uuid

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path('/work/asciibase64/Llama-3.2-1B-training')
MARKER = '# llama-1b-autoresearch-heartbeat'
STOP = PROJECT / '.heartbeat-disabled'

def cron_entry():
    return f'*/20 * * * * /usr/bin/python3 {shlex.quote(str(Path(__file__).resolve()))} run-once {MARKER}'

def read_cron():
    result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
    if result.returncode and 'no crontab' not in result.stderr.lower():
        raise RuntimeError('Cannot read crontab: ' + result.stderr.strip())
    return result.stdout

def update_cron(current, enabled):
    lines = [line for line in current.splitlines() if not line.rstrip().endswith(MARKER)]
    if enabled:
        lines.append(cron_entry())
    return '\n'.join(lines) + ('\n' if lines else '')

def atomic_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def run_once(check_only=False):
    if STOP.exists():
        print('Heartbeat disabled')
        return 0
    lease = PROJECT / '.heartbeat-maintenance.json'
    if not check_only and lease.exists():
        until = dt.datetime.fromisoformat(json.loads(lease.read_text())['until_utc'])
        if until > dt.datetime.now(dt.timezone.utc):
            print('Foreground implementation lease active; skipped')
            return 0
    directory = ROOT / 'runs/heartbeat'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Heartbeat already running; skipped')
            return 0
        codex = shutil.which('codex') or '/home/asciibase64/.local/bin/codex'
        if not Path(codex).is_file():
            raise RuntimeError('Codex CLI unavailable')
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        output = directory / (stamp + '.jsonl')
        errors = directory / (stamp + '.stderr')
        summary = directory / (stamp + '.md')
        prompt = ('Read HEARTBEAT.md and AGENTS.md. This is a heartbeat installation check only. '
                  'Do not mutate files or submit jobs. Confirm that research-state.yaml and findings.md '
                  'are readable, then summarize the current next action. Do not print credentials.'
                  if check_only else
                  'Continue authorized autoresearch in this repository. Read HEARTBEAT.md, AGENTS.md, '
                  'research-state.yaml and findings.md first. Execute one bounded heartbeat cycle; '
                  'diagnose and fix issues, account for jobs and advance ready stages. Do not wait in '
                  'a polling loop. Respect all resource, storage, split and run-budget rules. '
                  'Finish with an updated durable state and progress report.')
        command = [codex, 'exec', '--approve-for-me', '--cd', str(PROJECT), '--add-dir', str(ROOT),
                   '--ephemeral', '--color', 'never', '--json', '--output-last-message', str(summary), prompt]
        record = {'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'check_only': check_only,
                  'events': str(output), 'errors': str(errors), 'summary': str(summary), 'pid': os.getpid()}
        atomic_json(directory / 'last-run.json', record)
        environment = {**os.environ, 'PATH': '/home/asciibase64/.local/bin:/usr/local/bin:/usr/bin:/bin',
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        with output.open('w') as out, errors.open('w') as err:
            try:
                process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=out, stderr=err,
                                           start_new_session=True)
                code = process.wait(timeout=18 * 60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
        record.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), exit_code=code)
        atomic_json(directory / 'last-run.json', record)
        print(json.dumps(record))
        return code

def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['install', 'status', 'run-once', 'disable'])
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.action == 'run-once':
        return run_once(args.check_only)
    current = read_cron()
    if args.action in {'install', 'disable'}:
        enabled = args.action == 'install'
        if not enabled:
            STOP.touch()
        desired = update_cron(current, enabled)
        subprocess.run(['crontab', '-'], input=desired, text=True, check=True)
        if enabled:
            STOP.unlink(missing_ok=True)
        if read_cron() != desired:
            raise RuntimeError('Crontab verification failed')
    entries = [s for s in read_cron().splitlines() if s.rstrip().endswith(MARKER)]
    status = {'enabled': len(entries) == 1 and not STOP.exists(), 'entries': entries,
              'interval_minutes': 20, 'logs': str(ROOT / 'runs/heartbeat')}
    last = ROOT / 'runs/heartbeat/last-run.json'
    if last.exists():
        status['last_run'] = json.loads(last.read_text())
    print(json.dumps(status, indent=2))
    return 0

if __name__ == '__main__':
    sys.exit(main())
