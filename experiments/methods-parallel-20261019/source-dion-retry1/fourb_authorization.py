"""Exact user-approved formal14; one7h/56H allocation, fresh weights only."""
RUN_ID='raw4b-intradoc-normuon-20261017'
CONFIG_DIGEST='ed7d3b6317bd308ce76d1bcecc5db4c3332d8db2568ad56cdee1118bb0ab19bf'
def authorized(run_id,cfg,resume=False):
    from training_state import digest
    return run_id==RUN_ID and not resume and digest(cfg)==CONFIG_DIGEST
