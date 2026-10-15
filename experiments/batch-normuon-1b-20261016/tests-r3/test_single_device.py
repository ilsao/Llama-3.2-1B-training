from pathlib import Path
import sys
import unittest

LIVE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LIVE/'source-r3'))
from normuon_check import indexed_device


class Device:
    def __init__(self,kind,index=None):self.type,self.index=kind,index


class SingleDevice(unittest.TestCase):
    def test_only_unindexed_cuda_is_normalized(self):
        self.assertEqual(indexed_device(Device('cuda'),0).index,0)
        self.assertEqual(indexed_device(Device('cuda'),3).index,3)
        indexed=Device('cuda',2);self.assertIs(indexed_device(indexed,0),indexed)
        cpu=Device('cpu');self.assertIs(indexed_device(cpu,0),cpu)


if __name__=='__main__':unittest.main()
