"""Exact user-approved fresh formal12;one6h/48H allocation only."""
RUN_ID='raw3b-bs64-wsd40-20261016'
CONFIG_DIGEST='a1f147bde17a4d3ff050873568d40b2482263675e91338a68f8a550979e07da7'

def authorized(run_id,cfg,resume=False):
    from training_state import digest
    return run_id==RUN_ID and not resume and digest(cfg)==CONFIG_DIGEST
