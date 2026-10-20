"""Dependency-free identity and capacity contracts for the approved hidden spool."""
import hashlib
import os
from training_state import digest

RUN = 'kd3b-hidden8-20261019-seed42'
WORLD = 8
SHAPE = (8191, 3072)
BYTES = 8191 * 3072 * 2
SLOTS = 2048
CAP = 10**12
assert (SLOTS + 1) * WORLD * BYTES < CAP


def batch_digest(batch):
    h = hashlib.sha256()
    for key in ('input_ids', 'labels', 'document_ids', 'attention_mask'):
        h.update(key.encode())
        h.update(batch[key].numpy().tobytes())
    return h.hexdigest()


def identity(key, rank, batch, cfg, provenance):
    return dict(run_id=RUN, key=key, row=key*WORLD+rank, rank=rank,
        batch_sha256=batch_digest(batch), config_digest=digest(cfg),
        preparation_fingerprint=provenance['fingerprint'],
        teacher_manifest_sha256=cfg['parallel_method']['teacher']['manifest_sha256'],
        shape=list(SHAPE), dtype='bfloat16', representation='final_rmsnorm_hidden_excluding_last')


def ready_record(cfg, provenance):
    return dict(run_id=RUN, config_digest=digest(cfg),
        preparation_fingerprint=provenance['fingerprint'], world_size=WORLD,
        node=os.environ['SLURMD_NODENAME'], job_id=os.environ['SLURM_JOB_ID'],
        start_epoch=int(os.environ['TRAIN_START_EPOCH']),
        end_epoch=int(os.environ['TRAIN_END_EPOCH']))


def validate_peers(peers):
    left, right = peers['producer'], peers['consumer']
    for key in ('run_id', 'config_digest', 'preparation_fingerprint', 'world_size'):
        if left[key] != right[key]:
            raise ValueError('Teacher/student READY mismatch: '+key)
    if left['node'] == right['node'] or left['job_id'] == right['job_id']:
        raise ValueError('Require separate eight-GPU allocations')
    return min(left['end_epoch'], right['end_epoch'])
