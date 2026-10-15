"""Exact fresh formal10 authorization, requested full-SOTA3B comparison."""
RUN_ID='raw3b-sota-20261015'
CONFIG_DIGEST='bbe8b4c62678fbe64bbda66452752f41cac9e89b2860fc7166ee7e1b1e7285b7'

def authorized(run_id,cfg,resume=False):
    from training_state import digest
    return run_id==RUN_ID and not resume and digest(cfg)==CONFIG_DIGEST
