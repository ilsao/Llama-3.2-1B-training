"""Production integration of the validated frozen-teacher online pipeline."""
import os
import time
from pathlib import Path

import torch

from training_state import TokenJournal, atomic_json, file_digest, read_json
from online_pipeline import OnlineTeacherPipeline, attach_pipeline_loss


class PipelineRuntime:
    def __init__(self, cfg, run, rows, evaluation, provenance, accelerator):
        import train
        from checks import run_checks
        from transformers import AutoModelForCausalLM
        import teacher_attention

        self.cfg, self.run, self.accelerator = cfg, run, accelerator
        self.transaction = None
        self.next_key = 0
        self.batches = []
        self.ticket = None
        self.journal = None
        self.teacher = None
        self.pipe = None
        if cfg['parallel_method']['pipeline'] != dict(slots=2, overlap=True, scope='within_optimizer_window'):
            raise ValueError('Unvalidated pipeline configuration')
        train.startup_note('pipeline_preflight_start')
        # Every rank reads real first/last target rows before model initialization.
        for index in (accelerator.process_index,
                      cfg['target_training_tokens']//8192-8+accelerator.process_index):
            b = train.collate_rows([rows[index]], cfg['sequence_length'])
            if int(b['attention_mask'].sum()) != 8192 or 'document_ids' not in b:
                raise ValueError('Real canonical loader preflight failed')
            if int((b['labels'][:, 1:] != -100).sum()) <= 0:
                raise ValueError('Missing valid labels')
        checks = run_checks(accelerator.device)
        checks['teacher_attention'] = teacher_attention.check(accelerator.device)
        checks['real_loader_first_last_on_all_ranks'] = True
        train.on_main(accelerator, lambda: atomic_json(run/'pipeline-preflight.json', checks))
        spec = cfg['parallel_method']['teacher']
        teacher_root = Path(spec['export'])
        def verify():
            path = teacher_root/'teacher-manifest.json'
            if file_digest(path) != spec['manifest_sha256']:
                raise ValueError('Teacher manifest hash mismatch')
            manifest = read_json(path)
            if (manifest['model_id'] != spec['model_id'] or manifest['revision'] != spec['revision']
                    or not manifest['equivalent_tokenization_verified']):
                raise ValueError('Teacher identity/tokenizer mismatch')
            for name, item in manifest['files'].items():
                f = teacher_root/name
                if f.stat().st_size != item['size'] or file_digest(f) != item['sha256']:
                    raise ValueError('Teacher artifact mismatch: '+name)
            atomic_json(run/'teacher-provenance.json', manifest)
            return manifest
        self.manifest = train.on_main(accelerator, verify)
        train.startup_note('teacher_verified_loading')
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_root, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation='sdpa').to(accelerator.device)
        self.teacher.requires_grad_(False).eval()
        self.teacher.config.use_cache = False
        teacher_attention.configure(self.teacher)
        train.startup_note('teacher_evaluation_start')
        previous = os.environ.get('EVAL_DOCUMENT_OUTPUT')
        os.environ['EVAL_DOCUMENT_OUTPUT'] = str(run/'teacher-selection-documents.json')
        started = time.monotonic()
        try:
            self.teacher_evaluation = train.evaluate(self.teacher, evaluation, cfg, accelerator)
        finally:
            if previous is None:
                os.environ.pop('EVAL_DOCUMENT_OUTPUT', None)
            else:
                os.environ['EVAL_DOCUMENT_OUTPUT'] = previous
        if self.teacher_evaluation['eval/prediction_tokens'] != 1479302:
            raise ValueError('Teacher fixed 1k evaluation membership/count differs')
        self.teacher_evaluation.update(seconds=time.monotonic()-started,
            model_id=spec['model_id'], revision=spec['revision'],
            preparation_fingerprint=provenance['fingerprint'],
            checkpoint_loaded_for_student_training=False,
            accounting='Held-out inference; separate from student and KD teacher training-input journals')
        train.on_main(accelerator, lambda: atomic_json(run/'teacher-evaluation.json', self.teacher_evaluation))
        train.startup_note('teacher_evaluation_ready')
        def journal():
            self.journal = TokenJournal(run/'teacher-input-journal.json', cfg['max_training_tokens'])
            return self.journal.data
        self.journal_data = train.on_main(accelerator, journal)

    def attach(self, student):
        attach_pipeline_loss(student, self.cfg['parallel_method'], self.manifest)
        # Teacher stays outside student module/DDP/optimizer/checkpoint registration.
        if {id(p) for p in student.parameters()} & {id(p) for p in self.teacher.parameters()}:
            raise ValueError('Teacher parameters leaked into student')
        self.pipe = OnlineTeacherPipeline(self.teacher, positions=8191,
            vocab_size=128256, chunk=1024, overlap=True)

    def reserve_window(self, inputs):
        from train import on_main
        self.boundary()
        self.transaction = on_main(self.accelerator, lambda: self.journal.reserve(inputs))
        self.journal_data = on_main(self.accelerator, lambda: self.journal.data)

    def begin_window(self, local):
        if self.transaction is None:
            raise ValueError('Teacher input reservation required before any forward')
        self.base_key = self.next_key
        self.counts = [int(b['attention_mask'].sum()) for b in local]
        self.batches = [{k: v.to(self.accelerator.device) for k, v in b.items()} for b in local]
        self.next_key += len(local)
        self.pipe.prefetch(self.base_key, self.batches[0], nonpadding_inputs=self.counts[0])

    def batch(self, index):
        self.ticket = self.pipe.acquire(self.base_key+index)
        if index+1 < len(self.batches):
            self.pipe.prefetch(self.base_key+index+1, self.batches[index+1],
                               nonpadding_inputs=self.counts[index+1])
        return {**self.batches[index], 'teacher_ticket': self.ticket}

    def release(self):
        self.pipe.release(self.ticket)
        self.ticket = None

    def confirm_window(self):
        from train import on_main
        if self.pipe.queue or self.ticket is not None:
            raise ValueError('Unconsumed teacher batch at optimizer boundary')
        on_main(self.accelerator, lambda: self.journal.confirm(self.transaction))
        self.journal_data = on_main(self.accelerator, lambda: self.journal.data)
        self.transaction = None
        self.batches = []
        if self.pipe.teacher_inputs*8 != self.journal_data['confirmed_tokens']:
            raise ValueError('Teacher journal disagrees with dense eight-rank consumed inputs')

    def boundary(self):
        if self.transaction is not None or self.ticket is not None or (self.pipe and self.pipe.queue):
            raise ValueError('Pipeline must be empty at evaluation/checkpoint boundary')

    def accounting(self):
        self.boundary()
        return dict(teacher_inputs_confirmed=self.journal_data['confirmed_tokens'],
                    teacher_inputs_charged=self.journal_data['charged_tokens'],
                    discarded_prefetch_inputs=0,
                    prefetch_scope='within reserved optimizer window; no cross-boundary teacher work')

    def close(self):
        self.boundary()
        return self.pipe.drain()
