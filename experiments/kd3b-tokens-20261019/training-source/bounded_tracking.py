"""Bound terminal W&B network waits; durable metrics always precede this call."""
import signal,time
from training_state import atomic_json

class TrackingTimeout(TimeoutError):pass

def bounded(call,receipt,seconds=120):
    def alarm(*_):raise TrackingTimeout('W&B completion exceeded '+str(seconds)+' seconds')
    old=signal.signal(signal.SIGALRM,alarm);started=time.monotonic()
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        call();result=dict(synced=True)
    except Exception as error:
        result=dict(synced=False,error_type=type(error).__name__,error=str(error))
        print('W&B completion deferred; local results retained: '+str(error),flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    atomic_json(receipt,{**result,'seconds':time.monotonic()-started,'timeout_seconds':seconds})
    return result
