#!/usr/bin/env python3
"""Restore the recorded student's thinking prefix after verified upstream download."""
import hashlib
import json
import os
from pathlib import Path


def prepare(workspace):
    directory = workspace / 'model/DeepSeek-R1-Distill-Qwen-1.5B'
    data = json.loads((directory / 'tokenizer_config.upstream.json').read_text())
    old = "{{'<｜Assistant｜>'}}{% endif %}"
    new = r"{{'<｜Assistant｜><think>\n'}}{% endif %}"
    if data['chat_template'].count(old) != 1:
        raise ValueError('unexpected upstream student generation prompt')
    data['chat_template'] = data['chat_template'].replace(old, new)
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if digest != '0d6403c11a37d8b78a1c8ac02c408b3239e34f7d196855eb5ec84bc0385a4dc4':
        raise ValueError('student tokenizer config differs from the recorded configuration')
    destination = directory / 'tokenizer_config.json'
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print('Prepared student thinking chat template; canonical SHA-256:', digest)


if __name__ == '__main__':
    prepare(Path(os.environ['AUTORLBENCH_WORKSPACE']))
