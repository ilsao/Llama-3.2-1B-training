"""Read-only reuse of the canonical, already-completed 6B tail overlay."""
from bisect import bisect_right
from pathlib import Path
from bootstrap import HERE
from contracts import TARGET
from training_state import DEFAULT_ROOT,read_json,file_digest,atomic_json
OLD=HERE.parent/'raw6b-normuon-adam2x-20261018'
MANIFEST=DEFAULT_ROOT/'runs/data-overlays/raw6b-normuon-adam2x-20261018/manifest.json'


def verify_manifest():
    for name,sha in read_json(OLD/'raw-metadata-seal.json').items():
        if file_digest(name)!=sha:raise ValueError('Canonical raw metadata changed: '+name)
    m=read_json(MANIFEST);prefix=read_json(OLD/'raw-prefix.json')['files']
    if (m['run_id']!='raw6b-normuon-adam2x-20261018' or m['target_tokens']!=TARGET
        or not m['validated'] or m['train_files'][:-1]!=prefix[:-1]
        or m['original_last_shard']!=prefix[-1]
        or sum(x['rows'] for x in m['train_files'])*8192!=TARGET):
        raise ValueError('Invalid canonical6B overlay')
    return dict(manifest=str(MANIFEST),manifest_sha256=file_digest(MANIFEST),
                target_tokens=TARGET,files=m['train_files'],read_only=True)


class TailDataset:
    def __init__(self, receipt):
        self.files=receipt['files'];self.ends=[];n=0
        for f in self.files:n+=f['rows'];self.ends.append(n)
        self.cache={};self.verified={}
        assert n*8192==TARGET

    def __len__(self):return self.ends[-1]

    def shard(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        return bisect_right(self.ends,index)

    def verify_indices(self,indices):
        # Called only on rank0 BEFORE peers can read/forward these windows.
        from dataset import check_artifact
        for s in sorted({self.shard(i) for i in indices}):
            p=Path(self.files[s]['path']);st=p.stat();signature=(st.st_size,st.st_mtime_ns,st.st_ctime_ns)
            if s in self.verified and self.verified[s]!=signature:raise ValueError('Immutable shard changed')
            if s not in self.verified:
                check_artifact(self.files[s],True);self.verified[s]=signature
        return True

    def __getitem__(self,index):
        s=self.shard(index)
        if s not in self.cache:
            import datasets
            from dataset import check_artifact
            check_artifact(self.files[s],False)
            self.cache[s]=datasets.Dataset.from_file(self.files[s]['path'])
            for k in list(self.cache):
                if k not in (s,s-1):del self.cache[k]
        return self.cache[s][index-(self.ends[s-1] if s else 0)]

    def commit_receipt(self,path,cursor):
        n=bisect_right(self.ends,max(0,cursor-1))+1 if cursor else 0
        atomic_json(path,dict(cursor=cursor,files=self.files[:n],manifest=str(MANIFEST),manifest_sha256=file_digest(MANIFEST)))

