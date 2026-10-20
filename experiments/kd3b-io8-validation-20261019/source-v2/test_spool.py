import tempfile
import time
import unittest
import threading
from pathlib import Path
from spool import Spool, atomic_json, wait_for_roles


class ProtocolTests(unittest.TestCase):
    def test_own_rank0_publication_can_lag_external_peer(self):
        # Reproduce the failed8-rank startup: producer READY already exists,
        # while consumer nonzero ranks arrive before consumer rank0 publishes.
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for role in ('producer', 'consumer'):
                (root/role).mkdir()
            atomic_json(root/'producer/ready.json', dict(node='teacher-node'))
            def publish_late():
                time.sleep(.03)
                atomic_json(root/'consumer/ready.json', dict(node='student-node'))
            worker = threading.Thread(target=publish_late)
            worker.start()
            try:
                result = wait_for_roles(root, ('producer', 'consumer'), time.time()+2)
                self.assertEqual(result['consumer']['node'], 'student-node')
                self.assertNotEqual(result['producer']['node'], result['consumer']['node'])
            finally:
                worker.join()

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
