"""Run demo checks from any working directory; preserve logs and nonzero failures."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--web', action='store_true', help='Also run Chrome browser checks (requires Node + playwright)')
    parser.add_argument('--mobile', action='store_true', help='Also run Chromium/WebKit touch and mobile layout checks')
    args = parser.parse_args()
    output = ROOT / 'tests/results' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    commands = [
        ('unit_integration', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']),
        ('backend', [sys.executable, 'tests/test_backend.py']),
        ('stress_edge', [sys.executable, 'tests/stress_and_edge_test.py']),
    ]
    if args.web:
        commands.append(('browser', [shutil.which('node') or 'node', 'tests/frontend_demo.cjs']))
    if args.mobile:
        commands.append(('mobile_browsers', [shutil.which('node') or 'node', 'tests/mobile_browsers.cjs']))
    results = {}
    for name, command in commands:
        print(f'Running {name}...', flush=True)
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'SIC_TEST_OUTPUT': str(output)})
                results[name] = result.returncode
            except OSError as error:
                log.write(str(error))
                results[name] = 1
        print(f'{name}: {"PASS" if results[name] == 0 else "FAIL"}', flush=True)
    (output / 'summary.json').write_text(json.dumps({'passed': all(code == 0 for code in results.values()),
        'exit_codes': results, 'browser_requested': args.web, 'mobile_requested': args.mobile}, indent=2) + '\n', encoding='utf-8')
    print(f'Logs: {output}', flush=True)
    return 0 if all(code == 0 for code in results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
