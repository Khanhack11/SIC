import copy
import json
from pathlib import Path
import unittest

from ml.field_release import decide


class FieldReleaseTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        evidence = root / 'docs/final_release/mobile_field'
        self.baseline = json.loads((evidence/'baseline_validation.json').read_text())
        self.candidate = json.loads((evidence/'candidate_validation.json').read_text())
        self.criteria = json.loads((root/'docs/FIELD_RELEASE_CRITERIA.json').read_text())

    def test_recorded_candidate_remains_rejected_even_with_f1_improvement(self):
        result = decide(self.baseline, self.candidate, self.criteria)
        self.assertTrue(result['checks']['field_f1_gain']['passed'])
        self.assertFalse(result['checks']['field_score_gate_accuracy']['passed'])
        self.assertFalse(result['all_passed'])
        self.candidate['domains']['PlantDoc']['score_gate_accuracy'] = None
        self.assertFalse(decide(self.baseline, self.candidate, self.criteria)['all_passed'])

    def test_test_reports_or_different_data_cannot_drive_selection(self):
        for key, value in [('split', 'test'), ('records_sha256', 'changed')]:
            candidate = copy.deepcopy(self.candidate); candidate[key] = value
            with self.assertRaises(ValueError): decide(self.baseline, candidate, self.criteria)
