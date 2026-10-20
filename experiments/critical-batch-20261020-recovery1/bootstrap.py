"""Use the sealed, already-tested KD dependencies without modifying them."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
SOURCE=HERE.parent/'kd3b-hidden8-20261019/source'
FUSED=HERE.parent/'kd6b-lossless-20261020'
ACCEL=HERE.parent/'kd6b-lossless2-20261020'
sys.path[:0]=[str(HERE),str(HERE.parent/'kd6b-r5-20261020'),str(ACCEL),str(FUSED),str(SOURCE)]

