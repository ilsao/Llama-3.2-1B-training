"""Exact fresh formal9/10 authorization for the user-requested matched3B pair."""
RUNS={'raw-filtered-3b-20261015-raw':9}
CANCELLED_RUNS={'raw-filtered-3b-20261015-filtered':10}
CONFIG_DIGEST='33af588e388a6246e0943f0e69ba55986127d1a33d0c10bb28f88a8be7459359'

def authorized(run_id,config,resume=False):
    from training_state import digest
    return run_id in RUNS and not resume and digest(config)==CONFIG_DIGEST
