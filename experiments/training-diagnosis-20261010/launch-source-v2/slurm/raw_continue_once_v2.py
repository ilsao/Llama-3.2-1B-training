"""Diagnosed raw-producer lifecycle repair; original authorization stays immutable."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import filtering_parallel
from training_state import read_json,file_digest


class ManagedBatches(filtering_parallel.OrderedBatches):
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


if __name__=='__main__':
    receipt=Path(sys.argv[1]);recovery=read_json(receipt.parent/'producer-recovery.json')
    for name,expected in recovery['overlay_hashes'].items():
        if file_digest(Path(name))!=expected:raise ValueError('Raw recovery overlay changed')
    filtering_parallel.OrderedBatches=ManagedBatches
    from slurm.raw_continue_once import main
    sys.exit(main(receipt))
