#!/usr/bin/env python3
"""Reconstruct the recorded prompt content from pinned public source files."""
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def semantic_digest(rows):
    encoded = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare(workspace):
    data = workspace / 'datasets'
    rows = pq.read_table(data / '_source/dapo-processed.parquet').to_pylist()
    for row in rows:
        for message in row['prompt']:
            message['content'] = message['content'].replace('\\boxed{{}}', '\\boxed{}')
    if len(rows) != 17917 or semantic_digest(rows) != 'ac231cd004d8fd9f6299003fc8e2f976b09e8b1633278830f37f7113623c7c8e':
        raise ValueError('DAPO row content/order does not match the recorded run')
    pq.write_table(pa.Table.from_pylist(rows), data / 'dapo-math-17k-processed.parquet')

    source = [json.loads(line) for line in (data / '_source/aime2026.jsonl').read_text().splitlines()]
    suffix = ' Please reason step by step, and put your final answer within \\boxed{}.'
    rows = [
        {'prompt': [{'content': row['problem'].strip() + suffix, 'role': 'user'}],
         'data_source': 'AIME26', 'ability': 'math',
         'reward_model': {'ground_truth': str(row['answer']), 'style': 'rule'},
         'extra_info': {'index': f'AIME26-{index}', 'split': 'test'}}
        for index, row in enumerate(source)
    ]
    if len(rows) != 30 or semantic_digest(rows) != '007516e89620f83fee69dac2c806401f4e4c79cfc64f0a847a96e388358aef0e':
        raise ValueError('AIME26 row content/order does not match the recorded run')
    out = data / 'test_data/AIME26/test.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out)
    print('Prepared DAPO (17917) and AIME26 (30): canonical row hashes match the recorded data.')


if __name__ == '__main__':
    prepare(Path(os.environ['PTDBENCH_WORKSPACE']))
