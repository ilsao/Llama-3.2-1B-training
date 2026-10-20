"""Exact user-approved formal16; one7h/56H allocation, fresh weights only."""
RUN_ID='raw4b-normuon-adam2x-step30-20261018'
CONFIG_DIGEST='10dbaeef65171439aa652dad1ded13c5cd6deeac0f041e31f6af0e499747e2ad'
def authorized(run_id,cfg,resume=False):
    from training_state import digest
    return run_id==RUN_ID and not resume and digest(cfg)==CONFIG_DIGEST
