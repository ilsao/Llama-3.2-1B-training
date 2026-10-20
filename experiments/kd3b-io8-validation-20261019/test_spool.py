import tempfile
import time
import unittest
from pathlib import Path
from spool import Spool, atomic_json


class ProtocolTests(unittest.TestCase):
    def test_round_trip_and_ack_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            q = Spool(root, 0, 'logits16', 16, time.time()+1)
            data = bytes(range(16)); identity = {'row': 20, 'teacher': 'sealed'}
            q.publish(0, memoryview(data), identity)
            target = bytearray(16)
            q.read_into(0, memoryview(target), identity)
            self.assertEqual(bytes(target), data)
            with self.assertRaises(ValueError):
                q.read_into(0, memoryview(target), dict(identity, row=21))
            with self.assertRaises(ValueError):
                q.publish(0, memoryview(data), identity)
            q.acknowledge(0)
            self.assertTrue(q.path(0, 'ack').exists())
            self.assertFalse(q.path(0, 'bin').exists())

    def test_backpressure_and_partial_not_ready(self):
        with tempfile.TemporaryDirectory() as root:
            q = Spool(root, 0, 'hidden64', 4, time.time()+.03)
            q.publish(0, memoryview(b'1234'), {})
            q.publish(1, memoryview(b'5678'), {})
            with self.assertRaises(TimeoutError):
                q.publish(2, memoryview(b'abcd'), {})
            self.assertFalse(q.path(2, 'bin').exists())
            q.path(3, 'partial').write_bytes(b'0000')
            with self.assertRaises(TimeoutError):
                q.read_into(3, memoryview(bytearray(4)), {})

    def test_capacity_and_peer_failure(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                Spool(root, 0, 'oversize', 10**12, time.time()+1)
            q = Spool(root, 0, 'small', 1, time.time()+1)
            atomic_json(Path(root)/'abort-producer.json', {'reason': 'test'})
            with self.assertRaises(RuntimeError):
                q.read_into(0, memoryview(bytearray(1)), {})


if __name__ == '__main__':
    unittest.main()
