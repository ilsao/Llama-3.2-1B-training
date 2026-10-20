"""Metadata planner tests; no native libraries or worker launches on login."""
import copy
import random
from unittest import TestCase
import dataset as pipeline
from packing_parallel import training_plans,task_input


class PlannerTests(TestCase):
    def plans(self,docs,length,shard_rows,cursor=(0,0),carry=None):
        carry=carry or {}
        def metadata():
            for i in range(cursor[1],len(docs)):
                yield i,len(docs[i]),[0,i],[1,0] if i+1==len(docs) else [0,i+1]
        def decode(refs):
            tokens=[];labels=[];spans=[]
            for ref in refs:
                start=ref['token_offset'];count=ref['length']
                if ref['location'] is None:
                    piece=carry['tokens'][start:start+count];targets=carry['labels'][start:start+count]
                else:
                    piece=docs[ref['location'][1]][start:start+count]
                    targets=([-100]+piece[1:]) if start==0 else list(piece)
                tokens+=piece;labels+=targets
                spans.append({k:ref[k] for k in ('row_index','document_start','sequence_start','length')})
            return dict(tokens=tokens,labels=labels,spans=spans)
        return list(training_plans([{}],cursor,carry,length,shard_rows,metadata(),decode))

    def test_exact_boundaries_carry_masks_and_long_documents(self):
        rng=random.Random(42)
        for length in (4,8,19):
            for sizes in ([2,2,2,2],[length,2,length*12+3,2],
                          [rng.randrange(2,length*4) for _ in range(50)]):
                docs=[[1]+list(range(100+i,100+i+size-2))+[9] for i,size in enumerate(sizes)]
                plans=self.plans(docs,length,3);actual=[];last=0
                for plan in plans:
                    start=plan['start'][1] if plan['start'][0]==0 else len(docs)
                    end=plan['end'][1] if plan['end'][0]==0 else len(docs)
                    packer=pipeline.TrainingPacker(length,9,copy.deepcopy(plan['carry']));rows=[]
                    for i in range(start,end):rows.extend(packer.push(i,docs[i]))
                    if plan['final']:rows.extend(packer.finish())
                    self.assertEqual(len(rows),plan['expected_rows']);self.assertEqual(packer.state(),plan['carry_after'])
                    actual+=rows;last=end
                expected=list(pipeline.expected_sequences(((dict(row_index=i,tokens=t),None) for i,t in enumerate(docs)),length,9))
                self.assertEqual(actual,expected);self.assertEqual(last,len(docs))

    def test_resume_from_legacy_cross_document_carry(self):
        docs=[[1,2,3,9],[1,4,9],[1,5,6,7,9],[1,8,9]]
        packer=pipeline.TrainingPacker(8,9);prefix=[]
        for i in range(3):prefix+=list(packer.push(i,docs[i]))
        carry=copy.deepcopy(packer.state());plans=self.plans(docs,8,2,(0,3),carry)
        plan=plans[0];resumed=pipeline.TrainingPacker(8,9,copy.deepcopy(plan['carry']))
        actual=prefix+list(resumed.push(3,docs[3]))+list(resumed.finish())
        expected=list(pipeline.expected_sequences(((dict(row_index=i,tokens=t),None) for i,t in enumerate(docs)),8,9))
        self.assertEqual(actual,expected);self.assertEqual(resumed.state(),plan['carry_after'])

    def test_empty_remaining_input_finishes_saved_carry(self):
        carry=dict(tokens=[2,9],labels=[2,9],spans=[dict(row_index=7,document_start=4,sequence_start=0,length=2)])
        plans=list(training_plans([], [0,0],carry,8,2,metadata=[],decoder=lambda refs:carry))
        self.assertEqual(len(plans),1);self.assertTrue(plans[0]['final']);self.assertEqual(plans[0]['expected_rows'],1)
        self.assertEqual(task_input([],plans[0])['files'],[])
