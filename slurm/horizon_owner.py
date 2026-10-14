"""Shared submission ownership across host-local cron installations."""
import hashlib
import json
from pathlib import Path
import socket

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO/'experiments/horizon-wsd-20261014/submission-owner.json'


def require_owner(script, action):
    # Existing allocations may finish read-only checkpoint verification using
    # their original controller; all submissions/state/cron mutations need owner.
    if action in ('check', 'verify-job'):
        return True
    if not REGISTRY.exists():
        print('No active shared campaign owner; no mutation or submission.')
        return False
    owner = json.loads(REGISTRY.read_text())
    script = Path(script).resolve()
    if (str(script) != owner['script'] or
            hashlib.sha256(script.read_bytes()).hexdigest() != owner['sha256'] or
            socket.gethostname().split('.')[0] != owner['hostname']):
        print('Superseded controller or different login host; no mutation or submission. Owner: '
              + owner['script'] + ' on ' + owner['hostname'])
        return False
    return True
