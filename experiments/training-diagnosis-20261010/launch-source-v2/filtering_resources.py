"""Execution-only CPU override; immutable preparation configuration stays intact."""
import os
from preparation_parallel import worker_count as original_worker_count


def worker_count(config, *, scoring=False):
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    if cpus <= 8:
        return original_worker_count(config, scoring=scoring)
    expanded_train=(cpus,os.environ.get('FILTERING_TRAIN_WORKERS')) in ((12,'11'),(16,'15'))
    expanded_eval=cpus==12 and os.environ.get('FILTERING_EVAL_WORKERS')=='11'
    if not (expanded_train or expanded_eval):
        raise ValueError('Expanded CPUs require the explicit training-tokenization override')
    requested = config.get('score_workers' if scoring else 'preprocessing_workers', 1)
    if type(requested) is not int or not 1 <= requested <= 7:
        raise ValueError('Original preparation workers must remain1..7')
    return requested


def evaluation_worker_count(config):
    default=worker_count(config)
    requested=int(os.environ.get('FILTERING_EVAL_WORKERS','7'))
    if requested==7:return default
    if requested!=11 or int(os.environ.get('SLURM_CPUS_PER_TASK','1'))!=12:
        raise ValueError('11 evaluation workers require exactly12 allocated CPUs')
    return requested


def training_worker_count(config):
    default = worker_count(config)
    if os.environ.get('FILTERING_TRAIN_WORKERS', '7') == '7':
        return default
    requested=int(os.environ.get('FILTERING_TRAIN_WORKERS','7'))
    if (int(os.environ.get('SLURM_CPUS_PER_TASK','1')),requested) not in ((12,11),(16,15)):
        raise ValueError('Expanded training workers require matching allocated CPUs')
    return requested
