"""Exact user-approved fresh formal13;one6h/48H allocation only."""
RUN_ID='raw3b-bs64-stepwsd40-20261016'
CONFIG_DIGEST='a6084bbce35936772098e364b2c1bab5affcfe5ae3b979a77af2be32c81daf90'

def authorized(run_id,cfg,resume=False):
    from training_state import digest
    return run_id==RUN_ID and not resume and digest(cfg)==CONFIG_DIGEST
