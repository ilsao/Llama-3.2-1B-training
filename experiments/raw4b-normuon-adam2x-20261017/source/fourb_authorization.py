"""Exact user-approved formal15; one7h/56H allocation, fresh weights only."""
RUN_ID='raw4b-normuon-adam2x-20261017'
CONFIG_DIGEST='3f12b71e61585347c61a17ca4e369e2a1abdc8eb41c3402685102ae4b4d9db2f'
def authorized(run_id,cfg,resume=False):
    from training_state import digest
    return run_id==RUN_ID and not resume and digest(cfg)==CONFIG_DIGEST
