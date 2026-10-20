"""Read an immutable per-run tail overlay while retaining canonical eval identity."""
from raw_baseline import ProgressiveDataset,validate_append
from raw6b_tail import MANIFEST
from training_state import read_json,file_digest,atomic_json
from ppl_contracts import TARGET,STUDY

class TailDataset(ProgressiveDataset):
    def __init__(self,prep,receipt,stop_epoch):
        from pathlib import Path
        self.prep,self.receipt,self.stop_epoch=Path(prep),Path(receipt),stop_epoch
        self.target_rows=TARGET//8192;self.files=[];self.ends=[];self.cache={}
        self.refresh()
        if self.receipt.exists():validate_append(read_json(self.receipt)['files'],self.files)
    def refresh(self):
        m=read_json(MANIFEST)
        if m['run_id']!=STUDY or m['target_tokens']!=TARGET or not m['validated']:raise ValueError('Wrong tail overlay')
        validate_append(self.files,m['train_files']);self.files=m['train_files'];self.ends=[];total=0
        for item in self.files:total+=item['rows'];self.ends.append(total)
        if total!=TARGET//8192:raise ValueError('Wrong6B row count')
        self.status='complete'


def record_tail_provenance(path):
    atomic_json(path,dict(manifest=str(MANIFEST),sha256=file_digest(MANIFEST),run_id=STUDY,target_tokens=TARGET))
