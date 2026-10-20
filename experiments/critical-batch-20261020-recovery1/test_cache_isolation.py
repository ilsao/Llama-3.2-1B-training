import unittest
from rank_entry import private_paths


class IsolationTests(unittest.TestCase):
    def test_every_rank_has_unique_directories(self):
        values=[private_paths('/work/asciibase64/test','123',str(i)) for i in range(8)]
        for key in values[0]:self.assertEqual(len({v[key] for v in values}),8)

    def test_jobs_cannot_share_directories(self):
        a=private_paths('/work/asciibase64/test','123','0')
        b=private_paths('/work/asciibase64/test','124','0')
        self.assertTrue(all(a[k]!=b[k] for k in a))

    def test_reject_malformed_rank_and_job(self):
        for job,rank in [('123','../0'),('','0'),('abc','0')]:
            with self.assertRaises(ValueError):private_paths('/work/asciibase64/test',job,rank)


if __name__=='__main__':unittest.main()
