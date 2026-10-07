"""Small, dependency-free logic checks; no models, downloads, or worker pools."""
import hashlib
import json
import pickle
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dataset as pipeline
from third_party.dclm_fasttext import classify_fasttext_hq_prob


class Classifier:
    def __init__(self, label, probability):
        self.label, self.probability = label, probability
        self.inputs = []

    def predict(self, text):
        self.inputs.append(text)
        return [self.label], [self.probability]


class OfficialScoringTests(unittest.TestCase):
    def test_hq_probability_and_line_joining(self):
        model = Classifier('__label__hq', 0.7)
        self.assertEqual(classify_fasttext_hq_prob(model, '  abc\ndef\r\nghi  '), 0.7)
        self.assertEqual(model.inputs, ['abc def ghi'])

    def test_cc_complement_is_not_rounded_or_clamped(self):
        model = Classifier('__label__cc', 0.8)
        self.assertAlmostEqual(classify_fasttext_hq_prob(model, 'text'), 0.2)
        model.probability = 1.000001
        self.assertLess(classify_fasttext_hq_prob(model, 'text'), 0)


class PolicyTests(unittest.TestCase):
    def test_global_document_count_fraction_and_small_populations(self):
        self.assertEqual(pipeline.top_count(19, 0.1), 1)
        self.assertEqual(pipeline.top_count(9, 0.1), 0)
        self.assertEqual(pipeline.top_count(101, 0.1), 10)
        with self.assertRaises(ValueError):
            pipeline.top_count(10, 0.2)

    def test_complete_document_budget_counts_special_tokens(self):
        self.assertTrue(pipeline.fits_document(7, 7))
        self.assertFalse(pipeline.fits_document(8, 7))
        self.assertFalse(pipeline.fits_document(2, 1))
        self.assertFalse(pipeline.fits_document(1, 10))
        # An oversized document need not prevent a later complete document.
        remaining, kept = 8, []
        for count in [6, 4, 2]:
            if pipeline.fits_document(count, remaining):
                kept.append(count)
                remaining -= count
        self.assertEqual(kept, [6, 2])
        self.assertEqual(remaining, 0)

    def test_source_boundaries_and_unknown_source_failure(self):
        ranges = [{'start': 0, 'end': 2, 'source': 'common_crawl'},
                  {'start': 2, 'end': 5, 'source': 'stack_edu'}]
        self.assertEqual(pipeline.source_at(ranges, [2, 5], 1), 'common_crawl')
        self.assertEqual(pipeline.source_at(ranges, [2, 5], 2), 'stack_edu')
        with self.assertRaises(ValueError):
            pipeline.source_at(ranges, [2, 5], 5)
        self.assertEqual(pipeline.source_for_path('data/common_crawl-art-0014/shard_1.jsonl.zst'), 'common_crawl')
        self.assertEqual(pipeline.source_for_path('data/stack_edu-python-0000/shard_1.jsonl.zst'), 'stack_edu')
        with self.assertRaises(ValueError):
            pipeline.source_for_path('data/unknown/shard.jsonl.zst')
        self.assertEqual(pipeline.repo_filename('hf://datasets/a/b@abc/data/finemath-x/shard.jsonl.zst'),
                         'data/finemath-x/shard.jsonl.zst')

    def test_configuration_hard_invariants(self):
        cfg = pipeline.read_json(pipeline.PROJECT / 'configs/dataset.json')
        pipeline.validate_config(cfg)
        for key, value in [('max_training_tokens', 6_000_000_001), ('sequence_length', 4096),
                           ('shuffle_seed', 1), ('evaluation_documents', 1),
                           ('common_crawl_keep_fraction', 0.2), ('classifier_revision', 'main')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                pipeline.validate_config({**cfg, key: value})

    def test_heavy_entrypoint_rejects_login_node(self):
        with patch.dict('os.environ', {}, clear=True), self.assertRaises(RuntimeError):
            pipeline.require_compute()


class PackingTests(unittest.TestCase):
    def test_cross_document_spans_structural_bos_and_real_eos_padding(self):
        packer = pipeline.TrainingPacker(4, 9)
        rows = list(packer.push(10, [1, 3, 9]))
        rows += list(packer.push(20, [1, 4, 9]))
        rows += list(packer.finish())
        self.assertEqual(rows[0]['input_ids'], [1, 3, 9, 1])
        self.assertEqual(rows[0]['labels'], [-100, 3, 9, -100])
        self.assertEqual(rows[1]['input_ids'], [4, 9, 9, 9])
        self.assertEqual(rows[1]['attention_mask'], [1, 1, 0, 0])
        self.assertEqual(rows[1]['labels'], [-100, 9, -100, -100])
        self.assertEqual(sum(sum(row['attention_mask']) for row in rows), 6)
        self.assertEqual(rows[1]['document_spans'], [
            {'row_index': 20, 'document_start': 1, 'sequence_start': 0, 'length': 2}])

    def test_long_document_offsets_and_no_dropped_training_inputs(self):
        tokens = list(range(13))
        packer = pipeline.TrainingPacker(4, 99)
        rows = list(packer.push(5, tokens)) + list(packer.finish())
        self.assertEqual([row['document_spans'][0]['document_start'] for row in rows], [0, 4, 8, 12])
        restored = [token for row in rows for token, mask in zip(row['input_ids'], row['attention_mask']) if mask]
        self.assertEqual(restored, tokens)
        self.assertTrue(all(row['labels'][0] == -100 for row in rows))

    def test_checkpoint_carry_matches_uninterrupted_packing(self):
        documents = [(5, [1, 2, 9]), (6, [1, 3, 4, 9]), (7, [1, 9])]
        reference = pipeline.TrainingPacker(4, 9)
        expected = [row for index, tokens in documents for row in reference.push(index, tokens)]
        expected += list(reference.finish())
        partial = pipeline.TrainingPacker(4, 9)
        result = list(partial.push(*documents[0]))
        # JSON round-trip reflects the actual checkpoint persistence format.
        carry = json.loads(json.dumps(partial.state()))
        resumed = pipeline.TrainingPacker(4, 9, carry)
        result += [row for index, tokens in documents[1:] for row in resumed.push(index, tokens)]
        result += list(resumed.finish())
        self.assertEqual(result, expected)

    def test_evaluation_targets_once_at_exact_and_uneven_boundaries(self):
        for size in [2, 4, 5, 7, 8, 13]:
            with self.subTest(size=size):
                tokens = list(range(size))
                doc = {'row_index': 8, 'tokens': tokens, 'evaluation_slot': 2}
                rows = list(pipeline.pack_evaluation_document(doc, 4, 99))
                targets = [label for row in rows for label in row['labels'][1:] if label != -100]
                self.assertEqual(targets, tokens[1:])
                self.assertEqual(len(targets), size - 1)
                self.assertTrue(all(len(row['input_ids']) == 4 for row in rows))
                self.assertTrue(all(row['evaluation_slot'] == 2 for row in rows))
                self.assertTrue(all(len(row['document_spans']) == 1 for row in rows))


class DatabaseTests(unittest.TestCase):
    def test_global_top_ties_other_sources_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'clean').mkdir()
            (root / 'score').mkdir()
            with sqlite3.connect(root / 'clean/documents.sqlite') as db:
                db.execute('CREATE TABLE winners (digest BLOB PRIMARY KEY, row_index INTEGER, position INTEGER, source TEXT)')
                for index in range(20):
                    db.execute('INSERT INTO winners VALUES (?, ?, ?, ?)',
                               (bytes([index]), index, 20 - index, 'common_crawl'))
                db.execute('INSERT INTO winners VALUES (?, ?, ?, ?)', (b'other', 21, 0, 'stack_edu'))
            with sqlite3.connect(root / 'score/scores.sqlite') as db:
                db.execute('CREATE TABLE scores (row_index INTEGER PRIMARY KEY, position INTEGER, score REAL)')
                # All tied: winner must be the two smallest shuffle positions.
                db.executemany('INSERT INTO scores VALUES (?, ?, ?)', [(i, 20 - i, 0.5) for i in range(20)])
            def directory(stage):
                path = root / stage
                path.mkdir(exist_ok=True)
                return path
            prep = SimpleNamespace(path=root, config={'common_crawl_keep_fraction': 0.1},
                   directory=directory, manifest=lambda stage: {'details': {'scored_documents': 20}})
            candidate_path = pipeline.build_candidates(prep)
            with sqlite3.connect(candidate_path) as db:
                expected = [(21, 0, 'stack_edu'), (19, 1, 'common_crawl'), (18, 2, 'common_crawl')]
                self.assertEqual(db.execute('SELECT * FROM candidates ORDER BY position').fetchall(), expected)
            # Completed materialization is idempotent and does not duplicate rows.
            pipeline.build_candidates(prep)
            with sqlite3.connect(candidate_path) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0], 3)

    def test_dedup_keeps_first_shuffled_not_first_physical_row(self):
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE winners (digest BLOB PRIMARY KEY, row_index INTEGER, position INTEGER, source TEXT)')
            digest = hashlib.sha256(b'same text').digest()
            for row, pos in [(0, 9), (1, 3), (2, 8)]:
                db.execute('INSERT INTO winners VALUES (?, ?, ?, ?) '
                           'ON CONFLICT(digest) DO UPDATE SET row_index=excluded.row_index, '
                           'position=excluded.position, source=excluded.source '
                           'WHERE excluded.position < winners.position', (digest, row, pos, 'common_crawl'))
            self.assertEqual(db.execute('SELECT row_index, position FROM winners').fetchone(), (1, 3))

    def test_checksum_detects_same_size_modification(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'artifact'
            path.write_bytes(b'abcd')
            record = pipeline.artifact(path)
            pipeline.check_artifact(record, True)
            path.write_bytes(b'efgh')
            with self.assertRaises(ValueError):
                pipeline.check_artifact(record, True)


class TinyRaw:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def select(self, indices):
        return TinyRaw([self.rows[i] for i in indices])

    def iter(self, batch_size):
        for offset in range(0, len(self.rows), batch_size):
            rows = self.rows[offset:offset + batch_size]
            yield {key: [row[key] for row in rows] for key in ('id', 'text')}


class ResumeTests(unittest.TestCase):
    def test_actual_clean_resume_excludes_evaluation_before_dedup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            details = {'train_document_count': 4,
                       'ranges': [{'start': 0, 'end': 4, 'source': 'common_crawl'},
                                  {'start': 4, 'end': 6, 'source': 'stack_edu'}]}
            raw = TinyRaw([{'id': str(i), 'text': text} for i, text in
                           enumerate(['A', 'A', 'A', ' ', ' ', 'B'])])
            inverse = [5, 2, 0, 4, 1, 3]
            captured = {}
            def directory(stage):
                path = root / stage
                path.mkdir(exist_ok=True)
                return path
            def finish(stage, paths, stats):
                captured[stage] = stats
            prep = SimpleNamespace(path=root, config={'document_batch_size': 2}, directory=directory,
                                   manifest=lambda stage: {'details': details}, finish=finish)
            fake_np = SimpleNamespace(load=lambda *args, **kwargs: inverse)
            with patch.dict('sys.modules', {'numpy': fake_np}), patch.object(pipeline, 'load_raw', return_value=raw):
                def stop_after_committed_batch():
                    path = root / 'clean/documents.sqlite'
                    if path.exists():
                        with sqlite3.connect(path) as db:
                            if pipeline.progress(db, 'next_row', 0) == 2:
                                raise pipeline.StopRequested()
                with patch.object(pipeline, 'stop_if_requested', side_effect=stop_after_committed_batch):
                    with self.assertRaises(pipeline.StopRequested):
                        pipeline.stage_clean(prep)
                with sqlite3.connect(root / 'clean/documents.sqlite') as db:
                    self.assertEqual(pipeline.progress(db, 'next_row'), 2)
                pipeline.stage_clean(prep)
            with sqlite3.connect(root / 'clean/documents.sqlite') as db:
                self.assertEqual(db.execute('SELECT row_index, position FROM winners ORDER BY position').fetchall(),
                                 [(2, 0), (5, 3)])
                self.assertEqual(db.execute('SELECT row_index FROM empty').fetchall(), [(4,)])
            self.assertEqual(captured['clean']['eligible_documents'], 2)
            self.assertEqual(captured['clean']['empty_documents'], 1)
            self.assertEqual(captured['clean']['duplicate_documents'], 1)

    def test_actual_selection_whole_document_cap_and_interruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'select').mkdir()
            candidate_path = root / 'select/candidates.sqlite'
            with pipeline.database(candidate_path) as db:
                db.execute('CREATE TABLE candidates (row_index INTEGER PRIMARY KEY, position INTEGER, source TEXT)')
                db.executemany('INSERT INTO candidates VALUES (?, ?, ?)',
                               [(1, 0, 'stack_edu'), (3, 1, 'stack_edu'), (2, 2, 'stack_edu'),
                                (4, 3, 'stack_edu'), (6, 4, 'stack_edu')])
                pipeline.set_progress(db, 'common_crawl_kept', 0)
                pipeline.set_progress(db, 'boundary', None)
            raw = TinyRaw([{'id': str(i), 'text': text} for i, text in
                           enumerate(['E', 'A', 'CCCCC', 'BBB', 'D', 'EE', 'F'])])
            captured = {}
            class Tokenizer:
                bos_token_id, eos_token_id = 1, 9
                def encode(self, text, add_special_tokens):
                    assert add_special_tokens is False
                    return [ord(c) for c in text]
                def save_pretrained(self, directory):
                    directory.mkdir(exist_ok=True)
                    (directory / 'tokenizer.json').write_text('{}')
            def fixture(path, rows, schema):
                path.write_bytes(pickle.dumps(rows))
                return {**pipeline.artifact(path), 'rows': len(rows)}
            def directory(stage):
                path = root / stage
                path.mkdir(exist_ok=True)
                return path
            def finish(stage, paths, details):
                captured[stage] = details
            split = {'train_document_count': 5,
                     'ranges': [{'start': 0, 'end': 7, 'source': 'stack_edu'}]}
            prep = SimpleNamespace(path=root, config={'max_training_tokens': 10, 'document_shard_tokens': 4},
                    directory=directory, finish=finish, manifest=lambda stage: {'details': split})
            with patch.dict('sys.modules', {'numpy': SimpleNamespace(load=lambda *args, **kwargs: [0, 5])}), \
                    patch.object(pipeline, 'build_candidates', return_value=candidate_path), \
                    patch.object(pipeline, 'load_raw', return_value=raw), \
                    patch.object(pipeline, 'load_tokenizer', return_value=Tokenizer()), \
                    patch.object(pipeline, 'document_schema', return_value=None), \
                    patch.object(pipeline, 'write_arrow', side_effect=fixture):
                with patch.object(pipeline, 'stop_if_requested', side_effect=[None, pipeline.StopRequested()]):
                    with self.assertRaises(pipeline.StopRequested):
                        pipeline.stage_select(prep)
                checkpoint = pipeline.read_json(root / 'select/state.json')
                self.assertEqual(checkpoint['train_tokens'], 8)
                self.assertEqual(checkpoint['train_documents'], 2)
                pipeline.stage_select(prep)
            selected = captured['select']
            train_docs = [doc for file in selected['train_files'] for doc in pickle.loads(Path(file['path']).read_bytes())]
            eval_docs = [doc for file in selected['eval_files'] for doc in pickle.loads(Path(file['path']).read_bytes())]
            self.assertEqual([doc['row_index'] for doc in train_docs], [1, 3])
            self.assertEqual(sum(len(doc['tokens']) for doc in train_docs), 8)
            self.assertEqual(selected['budget_skipped_documents'], 3)
            self.assertEqual([doc['row_index'] for doc in eval_docs], [0, 5])
            self.assertEqual(selected['evaluation_tokens'], 7)
            self.assertTrue(all(doc['tokens'][0] == 1 and doc['tokens'][-1] == 9 for doc in train_docs + eval_docs))

    def test_actual_pack_resume_cursor_tokens_and_evaluation_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def fixture(path, rows, schema=None):
                path.write_text(json.dumps(rows))
                return {**pipeline.artifact(path), 'rows': len(rows)}
            def documents(files, cursor=(0, 0)):
                for shard in range(cursor[0], len(files)):
                    rows = json.loads(Path(files[shard]['path']).read_text())
                    start = cursor[1] if shard == cursor[0] else 0
                    for i in range(start, len(rows)):
                        after = (shard + 1, 0) if i + 1 == len(rows) else (shard, i + 1)
                        yield rows[i], after
            training = [{'row_index': i, 'tokens': tokens, 'evaluation_slot': -1} for i, tokens in
                        enumerate([list(range(12)), [1, 2, 9], [1, 3, 4, 5, 9]])]
            evaluation = [{'row_index': 10 + i, 'tokens': list(range(size)), 'evaluation_slot': i}
                          for i, size in enumerate([7, 5])]
            selected = {'train_files': [fixture(root / 'train-documents', training)],
                        'eval_files': [fixture(root / 'eval-documents', evaluation)],
                        'pad_token_id': 99, 'train_tokens': 20}
            captured = {}
            def directory(stage):
                path = root / stage
                path.mkdir(exist_ok=True)
                return path
            def finish(stage, paths, details):
                captured[stage] = details
            prep = SimpleNamespace(path=root, config={'sequence_length': 4, 'sequence_shard_rows': 2,
                                   'evaluation_subset_documents': 1}, directory=directory, finish=finish,
                                   manifest=lambda stage: {'details': selected})
            with patch.object(pipeline, 'write_arrow', side_effect=fixture), \
                    patch.object(pipeline, 'sequence_schema', return_value=None), \
                    patch.object(pipeline, 'iter_arrow', side_effect=documents):
                with patch.object(pipeline, 'stop_if_requested', side_effect=[None, pipeline.StopRequested()]):
                    with self.assertRaises(pipeline.StopRequested):
                        pipeline.stage_pack(prep)
                state = pipeline.read_json(root / 'pack/state.json')
                self.assertEqual(state['train_cursor'], [0, 1])
                self.assertEqual(state['training_tokens'], 12)
                pipeline.stage_pack(prep)
                actual = [row for row, _ in documents(captured['pack']['train_files'])]
                expected = list(pipeline.expected_sequences(documents(selected['train_files']), 4, 99))
                self.assertEqual(actual, expected)
            self.assertEqual(captured['pack']['training_tokens'], 20)
            self.assertEqual(captured['pack']['evaluation_prediction_tokens'], 10)
            self.assertEqual(captured['pack']['subset_prediction_tokens'], 6)


if __name__ == '__main__':
    unittest.main()
