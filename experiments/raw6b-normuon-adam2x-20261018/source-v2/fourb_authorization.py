"""Exact approved formal17: one10h/80H allocation and rounded-up6B cap."""
RUN_ID='raw6b-normuon-adam2x-20261018'
CONFIG_DIGEST='92b99288eb9afdfba6749bdd03058a42d27279f0b83765ec9424cb67da8c5f9b'
def authorized(run_id,cfg,resume=False):
    from training_state import digest
    return run_id==RUN_ID and not resume and digest(cfg)==CONFIG_DIGEST
