"""Host-only tests for the experiment helper. Does not load weights or use CUDA."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'tools/inkling2_deep_ab.py'
spec = importlib.util.spec_from_file_location('deep_ab', HELPER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DeepABTests(unittest.TestCase):
    def test_done(self):
        d = module.parse_done('DONE 2 100 50 4 length 1 2 0 8 10')
        self.assertEqual(d['decode_tps'], 500)
        self.assertEqual(d['gpu_hit_rate'], .8)

    def test_bad_done(self):
        with self.assertRaises(ValueError):
            module.parse_done('DONE 1')

    def test_fake_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            engine = d / 'engine'
            engine.write_text('#!' + sys.executable + '\n' + '''import sys
print('READY', flush=True)
for line in sys.stdin:
    if line.startswith('GEN '):
        n = len(line.split()[-1].strip().split(','))
        print('T 23', flush=True)
        print('T 42', flush=True)
        print(f'DONE 2 {n} 50 4 length 1 2 0 8 10', flush=True)
    elif line.startswith('QUIT'):
        break
''')
            engine.chmod(0o700)
            (d / 'prefill.ids').write_text('1,2\n3,4\n5\n')
            out = d / 'result.json'
            r = subprocess.run([sys.executable, str(HELPER), '--engine', str(engine),
                                '--model', str(d / 'fake.basalt'), '--fixtures', tmp,
                                '--depth', '32', '--tokens', '2', '--max-context', '1024',
                                '--out', str(out)], capture_output=True, text=True, timeout=10)
            self.assertEqual(r.returncode, 0, r.stderr)
            data = json.loads(out.read_text())
            self.assertEqual(data['token_ids'], [23, 42])
            self.assertEqual(data['prompt_tokens'], 35)
            self.assertEqual(data['gpu_hit_rate'], .8)
            self.assertIn('ttft_s', data)


if __name__ == '__main__':
    unittest.main()
