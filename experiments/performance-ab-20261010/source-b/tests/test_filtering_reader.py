"""Small stdlib bounded-reader and transactional score continuation tests."""
import contextlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dataset as pipeline
from filtering_reader import IndexedRaw
from slurm import filtering_entry as entry


class TinyDataset:
    def __init__(self, rows): self.rows=rows
    def __len__(self):return len(self.rows)
    def __iter__(self):return iter(self.rows)
    def select(self, indices):return TinyDataset([self.rows[i] for i in indices])
    def select_columns(self, columns):return TinyDataset([{k:r[k] for k in columns} for r in self.rows])


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.pieces=[TinyDataset([{'text':str(i),'id':i,'document_json':json.dumps({'text':str(i),'nested':[i]})} for i in indices]) for indices in [[0,1],[],[2,3,4],[5]]]
        self.index={'row_count':6,'files':[{'rows':len(d)} for d in self.pieces]}
        self.opens=[]
        def open_shard(i):self.opens.append(i);return self.pieces[i]
        self.raw=IndexedRaw(self.index,opener=open_shard)

    def test_global_positions_empty_shards_boundaries(self):
        self.assertEqual([self.raw[i]['id'] for i in range(6)],list(range(6)))
        self.assertNotIn(1,self.opens)
        self.assertEqual(self.raw[-1]['id'],5)
        self.assertLessEqual(self.raw.maximum_open,2)
        with self.assertRaises(IndexError):self.raw[6]

    def test_nonmonotonic_duplicates_column_and_full_field_equality(self):
        indices=[5,0,2,5,4,0]
        expected=[dict(self.pieces[0].rows[0]) if i==0 else self.raw[i] for i in indices]
        self.assertEqual(list(self.raw.select(indices)),expected)
        self.assertEqual(self.raw.select(indices)['text'],[str(i) for i in indices])
        self.assertEqual(self.raw[1:4]['id'],[1,2,3])
        self.assertLessEqual(len(self.raw.cache),2)

    def test_retains_recent_shards_and_fails_closed_on_bad_index(self):
        self.raw[0];self.raw[1]
        self.assertEqual(self.opens,[0])
        self.raw[2];self.raw[5]
        self.assertEqual(list(self.raw.cache),[2,3])
        self.raw.close();self.assertEqual(len(self.raw.cache),0)
        with self.assertRaises(ValueError):IndexedRaw({**self.index,'row_count':7})
        bad=IndexedRaw({'row_count':7,'files':[{'rows':7}]},opener=lambda _:self.pieces[0])
        with self.assertRaisesRegex(ValueError,'row count'):bad[0]

    def test_batch_guard_before_unbounded_data_loading(self):
        with self.assertRaises(ValueError):self.raw.select(range(2049))
        self.assertEqual(self.opens,[])
        with self.assertRaises(IndexError):self.raw.fetch([-1])


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name);(self.path/'clean').mkdir();(self.path/'score').mkdir()
        (self.path/'filtering-access').mkdir();pipeline.atomic_json(self.path/'filtering-access/validation.json',{'passed':True})
        self.model=self.path/'model';self.model.write_text('fixture')
        with sqlite3.connect(self.path/'clean/documents.sqlite') as db:
            db.execute('CREATE TABLE winners(row_index INTEGER PRIMARY KEY,position INTEGER,source TEXT)')
            db.executemany('INSERT INTO winners VALUES (?,?,?)',[(i,i*2,'common_crawl') for i in [1,2,4,9]])
        with pipeline.database(self.path/'score/scores.sqlite') as db:
            db.execute('CREATE TABLE scores(row_index INTEGER PRIMARY KEY,position INTEGER NOT NULL,score REAL NOT NULL)')
            db.executemany('INSERT INTO scores VALUES (?,?,?)',[(1,2,.1),(2,4,.2)])
            pipeline.set_progress(db,'last_row',2)
        self.finished=[]
        self.prep=SimpleNamespace(path=self.path,key='original',config={'score_workers':1,'document_batch_size':1},
            directory=lambda name:self.path/name,manifest=lambda name:{'details':{'source_counts':{'common_crawl':4}}},
            finish=lambda stage,files,details:self.finished.append(details))
        self.raw=IndexedRaw({'row_count':10,'files':[{'rows':10}]},opener=lambda _:TinyDataset([{'text':str(i)} for i in range(10)]))

    @contextlib.contextmanager
    def ordered(self,*args,**kwargs):
        yield lambda f,items:map(f,items)

    def patches(self, scorer):
        stack=contextlib.ExitStack()
        stack.enter_context(patch.dict('os.environ',{'SLURM_JOB_ID':'fixture','SLURM_CPUS_PER_TASK':'8'}))
        stack.enter_context(patch.object(entry,'identity',return_value={'original':True}))
        stack.enter_context(patch.object(entry,'ordered_executor',side_effect=self.ordered))
        stack.enter_context(patch.object(pipeline,'score_texts',side_effect=scorer))
        stack.enter_context(patch('builtins.print'))
        return stack

    def test_resume_preserves_saved_scores_and_counts_exactly_once(self):
        with self.patches(lambda texts:[float(t)/10 for t in texts]):entry.score(self.prep,self.raw,str(self.model))
        with sqlite3.connect(self.path/'score/scores.sqlite') as db:
            self.assertEqual(db.execute('SELECT row_index,score FROM scores ORDER BY row_index').fetchall(),[(1,.1),(2,.2),(4,.4),(9,.9)])
            self.assertEqual(pipeline.progress(db,'last_row'),9)
            self.assertEqual(pipeline.progress(db,'scored_documents'),4)
            self.assertEqual([r[2] for r in db.execute('PRAGMA index_info(score_ranking)')],['score','position'])
        progress=pipeline.read_json(self.path/'score/progress.json')
        self.assertEqual(progress['status'],'complete');self.assertEqual(progress['session_documents'],2)
        self.assertEqual(len(self.finished),1)

    def test_failed_batch_keeps_last_committed_cursor_then_exact_resume(self):
        def interrupted(texts):
            if texts==['9']:raise pipeline.StopRequested('pause')
            return [float(t)/10 for t in texts]
        with self.patches(interrupted),self.assertRaises(pipeline.StopRequested):entry.score(self.prep,self.raw,str(self.model))
        with sqlite3.connect(self.path/'score/scores.sqlite') as db:
            self.assertEqual(pipeline.progress(db,'last_row'),4)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM scores').fetchone()[0],3)
        self.assertEqual(pipeline.read_json(self.path/'score/progress.json')['status'],'paused')
        with self.patches(lambda texts:[float(t)/10 for t in texts]):entry.score(self.prep,self.raw,str(self.model))
        self.assertEqual(pipeline.read_json(self.path/'score/progress.json')['completed_documents'],4)
        self.assertEqual(len(self.finished),1)

    def test_bad_cursor_is_rejected_without_erasing_existing_rows(self):
        with pipeline.database(self.path/'score/scores.sqlite') as db:pipeline.set_progress(db,'last_row',1)
        with self.patches(lambda _:[]),self.assertRaisesRegex(ValueError,'cursor'):entry.score(self.prep,self.raw,str(self.model))
        with sqlite3.connect(self.path/'score/scores.sqlite') as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM scores').fetchone()[0],2)
        self.assertFalse(self.finished)


if __name__=='__main__':unittest.main()
