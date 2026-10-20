"""Exact user-approved fresh formal11, single6h/48H allocation."""
RUN_ID='raw3b-sota-wsd20-20261015'
CONFIG_DIGEST='efd9b2a00aa2d225d9bc6b84e433384093288ed8ed266417591b46f8f432b5a7'

def authorized(run_id,cfg,resume=False):
    from training_state import digest
    return run_id==RUN_ID and not resume and digest(cfg)==CONFIG_DIGEST
