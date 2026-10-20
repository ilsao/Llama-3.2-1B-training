"""Stdlib-only tests for ordered two-credit scheduling and canonical adapter."""
from types import SimpleNamespace
from unittest import TestCase
import filtering_parallel as parallel


class SchedulerTests(TestCase):
    def session(self,workers=3):
        session=object.__new__(parallel.OrderedBatches)
        session.count=workers;session.stop=SimpleNamespace(is_set=lambda:False)
        session.pending=[0]*workers;session.maximum=[0]*workers;session.metrics={}
        class Pipe:
            def __init__(self,number):self.number=number;self.tasks=[]
            def put(self,value,timeout=None):self.tasks.append(value)
            def get(self,timeout=None):
                seq,rows=self.tasks.pop(0)
                return seq,[dict(row_index=r[0]) for r in rows],dict(worker=self.number)
        session.inputs=[Pipe(i) for i in range(workers)];session.outputs=session.inputs
        session.processes=[SimpleNamespace(is_alive=lambda:True) for _ in range(workers)]
        return session

    def test_order_and_two_credit_bound_for_short_and_long_streams(self):
        for workers in (3,7,15):
            for count in (0,1,3,6,7,35,61):
                session=self.session(workers);items=[[(i,i,'source',-1)] for i in range(count)]
                result=list(session.ordered(items))
                self.assertEqual([d[0]['row_index'] for d in result],list(range(count)))
                self.assertTrue(all(v<=2 for v in session.maximum));self.assertEqual(sum(session.pending),0)

    def test_no_credit_return_until_consumer_advances(self):
        session=self.session();iterator=session.ordered([[(i,i,'source',-1)] for i in range(20)])
        next(iterator);self.assertEqual(session.pending,[2,2,2])
        self.assertEqual(sum(len(p.tasks) for p in session.inputs),5)
        next(iterator);self.assertEqual(session.pending,[2,2,2]);iterator.close()

    def test_rejects_oversize_batches(self):
        with self.assertRaisesRegex(ValueError,'batch'):list(self.session().ordered([[None]*33]))

    def test_prepared_document_identity_and_order_checked(self):
        doc=parallel.PreparedDocument(row_index=3,shuffle_position=4,source='s',evaluation_slot=-1,tokens=[1,2,9])
        result=list(parallel.prepared_documents(None,None,[(doc,3,4,'s',-1)],7));self.assertEqual(result,[dict(doc)])
        with self.assertRaisesRegex(ValueError,'identity'):list(parallel.prepared_documents(None,None,[(doc,3,5,'s',-1)],7))
        with self.assertRaises(ValueError):list(parallel.prepared_documents(None,None,[({},3,4,'s',-1)],7))
