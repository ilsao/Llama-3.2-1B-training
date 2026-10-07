"""Bounded, ordered CPU work. Importing this module starts no workers."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
import multiprocessing
import os


def worker_count(config, *, scoring=False):
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    if not 1 <= cpus <= 8:
        raise ValueError('Each job must allocate between 1 and 8 CPUs')
    requested = config.get('score_workers' if scoring else 'preprocessing_workers', 1)
    if type(requested) is not int or not 1 <= requested <= 7:
        raise ValueError('Use 1..7 workers, reserving a CPU for the coordinator')
    return min(requested, max(1, cpus - 1))


def configure_threads():
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def bounded_map(executor, function, items, limit, check=lambda: None):
    """Yield submission order, with at most limit inputs/results retained."""
    iterator, pending = iter(items), deque()
    try:
        for _ in range(limit):
            check()
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append(executor.submit(function, item))
        while pending:
            check()
            result = pending.popleft().result()
            check()
            yield result
            check()
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(function, item))
    finally:
        for future in pending:
            future.cancel()


@contextmanager
def ordered_executor(workers, *, initializer=None, initargs=(), threads=False, check=lambda: None):
    if workers == 1:
        if initializer:
            initializer(*initargs)
        def inline(function, items):
            for item in items:
                check()
                yield function(item)
        yield inline
        return
    configure_threads()
    cls = ThreadPoolExecutor if threads else ProcessPoolExecutor
    kwargs = {'max_workers': workers, 'initializer': initializer, 'initargs': initargs}
    if not threads:
        kwargs['mp_context'] = multiprocessing.get_context('spawn')
    executor = cls(**kwargs)
    try:
        def ordered(function, items):
            return bounded_map(executor, function, items, 2 * workers, check)
        yield ordered
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
