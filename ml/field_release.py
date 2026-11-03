"""Apply the predeclared validation gate without reading test data or replacing weights."""
import argparse
import hashlib
import json
from pathlib import Path


def decide(baseline, candidate, criteria):
    if baseline['split'] != 'validation' or candidate['split'] != 'validation':
        raise ValueError('Release selection requires validation reports, never test reports')
    if baseline['records_sha256'] != candidate['records_sha256']:
        raise ValueError('Models were evaluated with different manifests')
    if criteria['confidence_threshold'] != .75:
        raise ValueError('These evaluation reports use the fixed 0.75 score gate')
    field = candidate['domains']['PlantDoc']; village = candidate['domains']['PlantVillage']
    checks = {
        'field_f1_gain': {'value': 100*(field['known_macro_f1_present']-baseline['domains']['PlantDoc']['known_macro_f1_present']),
                          'minimum': criteria['field_known_macro_f1_min_gain_percentage_points'], 'unit': 'percentage points'},
        'village_f1_drop': {'value': 100*(baseline['domains']['PlantVillage']['known_macro_f1_present']-village['known_macro_f1_present']),
                            'maximum': criteria['plantvillage_known_macro_f1_max_drop_percentage_points'], 'unit': 'percentage points'},
        'field_score_gate_accuracy': {'value': field['score_gate_accuracy'], 'minimum': criteria['field_accepted_accuracy_min']},
        'field_score_gate_coverage': {'value': field['score_gate_coverage'], 'minimum': criteria['field_accepted_coverage_min']},
        'known_diseased_false_healthy': {'value': field['known_diseased_high_confidence_healthy_rate'],
                                        'maximum': criteria['known_diseased_field_high_confidence_healthy_rate_max']},
    }
    for check in checks.values():
        check['passed'] = check['value'] is not None and (
            check['value'] >= check['minimum'] if 'minimum' in check else check['value'] <= check['maximum'])
    passed = all(check['passed'] for check in checks.values())
    return {'selection': 'validation only', 'checks': checks, 'all_passed': passed,
            'decision': 'Validation criteria passed; no automatic deployment' if passed else 'Keep default V2; candidate remains experimental',
            'candidate_weights_sha256': candidate['weights_sha256'], 'test_used_for_selection': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('docs/benchmarks/quality/baseline_validation.json'))
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--criteria', type=Path, default=Path('docs/FIELD_RELEASE_CRITERIA.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve existing evidence; choose a new output')
    result = decide(*(json.loads(p.read_text()) for p in [args.baseline, args.candidate, args.criteria]))
    result['evidence_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [args.baseline, args.candidate, args.criteria]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result['decision'])
