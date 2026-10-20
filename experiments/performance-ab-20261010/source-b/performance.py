"""Opt-in acceleration; tensor operations are called only on allocated nodes."""
from collections import deque
from contextlib import contextmanager
import hashlib
import inspect
import os
from pathlib import Path


class TrainingPerformance:
    def __init__(self, cfg):
        self.settings = cfg.get('performance', {})
        self.enabled = any(self.settings.get(key, False) for key in
                           ('model_compile', 'muon_compile', 'async_transfer'))
        self.backend = cfg.get('attention_backend', 'sdpa')
        self.buffers = []
        self.copy_stream = None
        self.recent_steps = deque(maxlen=20)
        self.metadata = {'requested': self.settings, 'enabled': self.enabled,
                         'model_compiled': False, 'muon_core_compiled': False,
                         'async_transfer': self.settings.get('async_transfer', False)}

    def configure(self, model):
        if not self.enabled:
            return model
        import torch
        if not os.environ.get('SLURMD_NODENAME'):
            raise RuntimeError('Compile requires an allocated compute node')
        if self.settings.get('muon_compile'):
            import torch.optim._muon as native
            if torch.__version__.split('+')[0] != '2.9.1':
                raise RuntimeError('Muon compilation is pinned to PyTorch 2.9.1')
            original = native._zeropower_via_newtonschulz
            self.metadata['muon_core_source_sha256'] = hashlib.sha256(
                inspect.getsource(original).encode()).hexdigest()
            native._zeropower_via_newtonschulz = torch.compile(
                original, backend='inductor', dynamic=False, fullgraph=True,
                mode='max-autotune-no-cudagraphs')
            self.metadata['muon_core_compiled'] = True
        if self.settings.get('model_compile'):
            model.compile(backend='inductor', dynamic=False, fullgraph=False,
                          mode=self.settings['compile_mode'])
            self.metadata['model_compiled'] = True
        return model

    def stage(self, batches, device):
        """Bounded pinned buffers; reservations/counts use the original CPU masks."""
        import torch
        if self.settings.get('async_transfer') and self.copy_stream is None:
            self.copy_stream = torch.cuda.Stream(device=device)
        result = []
        for index, batch in enumerate(batches):
            dense = self.backend == 'flash_attention_3' and bool(batch['attention_mask'].all())
            values = {key: value for key, value in batch.items()
                      if not (self.enabled and dense and key == 'attention_mask')}
            if not self.settings.get('async_transfer'):
                result.append(({key: value.to(device) for key, value in values.items()}, None))
                continue
            if index >= len(self.buffers):
                self.buffers.append({})
            buffers = self.buffers[index]
            output = {}
            with torch.cuda.stream(self.copy_stream):
                for key, value in values.items():
                    if key not in buffers or buffers[key][0].shape != value.shape:
                        buffers[key] = (torch.empty_like(value, pin_memory=True),
                                        torch.empty_like(value, device=device))
                    host, gpu = buffers[key]
                    host.copy_(value)
                    gpu.copy_(host, non_blocking=True)
                    output[key] = gpu
                ready = torch.cuda.Event()
                ready.record(self.copy_stream)
            result.append((output, ready))
        return result

    def step_begin(self):
        if self.settings.get('model_compile'):
            import torch
            torch.compiler.cudagraph_mark_step_begin()

    @contextmanager
    def forward_context(self, model, accelerator, batch):
        raw = accelerator.unwrap_model(model)
        compiled = getattr(raw, '_compiled_call_impl', None)
        padded = self.backend == 'flash_attention_3' and 'attention_mask' in batch
        if self.enabled and padded:
            raw._compiled_call_impl = None
        try:
            yield
        finally:
            if self.enabled and padded:
                raw._compiled_call_impl = compiled

    def observe(self, seconds, previous):
        if not self.enabled:
            return max(previous, seconds)
        self.recent_steps.append(seconds)
        return max(self.recent_steps)

    def diagnostics(self):
        if not self.enabled:
            return self.metadata
        from torch._dynamo.utils import counters
        return {**self.metadata, 'dynamo_counters':
                {name: dict(values) for name, values in counters.items() if values}}


def eager_loss_model(model):
    # nn.Module.compile optimizes __call__; direct forward preserves Accelerate AMP
    # while keeping variable-length evaluation outside the compiled top-level call.
    return model.forward if getattr(model, '_compiled_call_impl', None) is not None else model
