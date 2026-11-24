#!/usr/bin/env python3
"""Measure greedy decode after a real long prompt, using Basalt's engine protocol.
No downloads or third-party Python dependencies. Run from the experiment checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def parse_done(line):
    f = line.split()
    if len(f) < 8 or f[0] != 'DONE':
        raise ValueError('invalid DONE line')
    result = dict(generated=int(f[1]), prompt=int(f[2]), prompt_ms=float(f[3]),
                  decode_ms=float(f[4]), finish=f[5], accepted=int(f[6]), offered=int(f[7]))
    if len(f) > 10:
        result.update(hits=int(f[9]), lookups=int(f[10]))
    result['decode_tps'] = result['generated'] * 1000 / result['decode_ms'] if result['decode_ms'] > 0 else None
    if result.get('lookups', 0):
        result['gpu_hit_rate'] = result['hits'] / result['lookups']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--fixtures', default='data/bench/inkling2')
    p.add_argument('--depth', type=int, default=32768)
    p.add_argument('--max-context', type=int, default=262144)
    p.add_argument('--kv', choices=['fp16', 'fp8', 'int8'], default='fp8')
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--spec', type=int, default=8)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if a.depth < 1 or a.tokens < 1 or a.depth + a.tokens + 256 > a.max_context:
        p.error('depth and tokens must be positive and fit with 256 tokens of headroom')
    parts = Path(a.fixtures, 'prefill.ids').read_text().splitlines()
    if len(parts) != 3:
        p.error('prefill.ids must have three lines')
    pre, body, post = [[int(x) for x in line.split(',') if x.strip()] for line in parts]
    if not body:
        p.error('empty prefill body')
    # Same exact prompt for both builds. A new engine process means no cached prefix.
    ids = pre + [body[i % len(body)] for i in range(a.depth)] + post
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(Path(a.engine).resolve()), '--serve', '--model', str(Path(a.model).resolve()),
           '--gpu', '0', '--kv', a.kv, '--max-context', str(a.max_context), '--spec', str(a.spec)]
    result = {'command': cmd, 'depth_body_tokens': a.depth, 'prompt_tokens': len(ids),
              'prompt_sha256': hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest(),
              'note': 'greedy repeated-filler prompt, not a coding-quality test'}
    with out.with_suffix('.engine.log').open('w') as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                text=True, bufsize=1)
        try:
            info = []
            for line in proc.stdout:
                info.append(line.rstrip())
                if line.startswith('READY'):
                    break
                if line.startswith('ERR'):
                    raise RuntimeError(line.strip())
            else:
                raise RuntimeError('engine exited before READY')
            result['startup'] = info
            start = time.monotonic()
            proc.stdin.write('GEN ' + str(a.tokens) + ' temperature=0 ' + ','.join(map(str, ids)) + '\n')
            proc.stdin.flush()
            tokens = []
            for line in proc.stdout:
                if line.startswith('T '):
                    if not tokens:
                        result['ttft_s'] = time.monotonic() - start
                    tokens.append(int(line.split()[1]))
                elif line.startswith('DONE'):
                    result.update(parse_done(line))
                    result['raw_done'] = line.rstrip()
                    break
                elif line.startswith('ERR'):
                    raise RuntimeError(line.strip())
            else:
                raise RuntimeError('engine exited before DONE')
            result['wall_s'] = time.monotonic() - start
            result['token_ids'] = tokens
            result['token_sha256'] = hashlib.sha256(','.join(map(str, tokens)).encode()).hexdigest()
            if len(tokens) != result['generated']:
                raise RuntimeError('token count does not match DONE')
            proc.stdin.write('QUIT\n')
            proc.stdin.flush()
            proc.wait(timeout=120)
            if proc.returncode:
                raise RuntimeError('engine exit ' + str(proc.returncode))
        except Exception as e:
            result['error'] = str(e)
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['token_ids', 'startup']}, indent=2))


if __name__ == '__main__':
    main()
