"""Dependency-free validation of seek index trust boundaries."""
import unittest
import dataset as pipeline
from filtering_batch_reader import validate_sidecar


class IndexTests(unittest.TestCase):
    def value(self):
        original=dict(path='/fixture/raw.arrow',size=100,sha256='raw',rows=3)
        identity=dict(preparation_fingerprint='original',split_sha256='split')
        value=dict(format_version=1,original=original,identity=identity,schema_sha256='schema',batches=[
            dict(offset=8,end_offset=40,start_row=0,rows=0),
            dict(offset=40,end_offset=60,start_row=0,rows=2),
            dict(offset=60,end_offset=92,start_row=2,rows=1)])
        value['content_sha256']=pipeline.fingerprint(value)
        return value,original,identity

    def test_accepts_empty_batches_and_exact_coverage(self):
        value,original,identity=self.value()
        self.assertIs(validate_sidecar(value,original,identity),value)

    def test_changed_identity_and_unsealed_mutation_rejected(self):
        value,original,identity=self.value()
        with self.assertRaises(ValueError):validate_sidecar(value,original,{'split_sha256':'other'})
        value['batches'][1]['rows']=1
        with self.assertRaisesRegex(ValueError,'checksum'):validate_sidecar(value,original,identity)

    def test_invalid_ranges_rejected_even_if_resealed(self):
        for key,val in [('start_row',1),('rows',-1),('offset',39),('end_offset',101)]:
            value,original,identity=self.value();value['batches'][1][key]=val
            value['content_sha256']=pipeline.fingerprint({k:v for k,v in value.items() if k!='content_sha256'})
            with self.subTest(key=key),self.assertRaises(ValueError):validate_sidecar(value,original,identity)
        value,original,identity=self.value();value['batches'][-1]['rows']=2
        value['content_sha256']=pipeline.fingerprint({k:v for k,v in value.items() if k!='content_sha256'})
        with self.assertRaisesRegex(ValueError,'row count'):validate_sidecar(value,original,identity)
