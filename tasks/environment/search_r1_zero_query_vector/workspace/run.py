"""Check or train a self-contained source workspace; never overwrite a run."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
SHARED = ROOT / 'shared'


def overrides(config, run, checkpoints):
    return [
        f'data.train_batch_size={config["batch_size"]}', 'data.val_batch_size=16',
        'data.max_start_length=512', 'data.max_prompt_length=3072',
        'data.max_response_length=256', 'data.max_obs_length=768',
        'actor_rollout_ref.rollout.n_agent=4', 'actor_rollout_ref.rollout.max_num_seqs=32',
        f'actor_rollout_ref.actor.ppo_mini_batch_size={config["batch_size"] * 4}',
        f'actor_rollout_ref.actor.optim.lr={config["learning_rate"]}',
        'actor_rollout_ref.actor.entropy_coeff=0.001',
        '+actor_rollout_ref.actor.checkpoint_dtype=bfloat16',
        '+actor_rollout_ref.rollout.engine_seed=42', '+trainer.seed=42',
        '+trainer.answer_only_reward=true', '+trainer.ray_num_cpus=8',
        '+trainer.ray_object_store_memory=2147483648',
        f'trainer.total_training_steps={config["steps"]}',
        f'trainer.total_epochs={math.ceil(config["steps"] / (1024 // config["batch_size"]))}',
        f'trainer.test_freq={config["eval_every"]}', 'trainer.val_before_train=true',
        f'trainer.save_freq={config["steps"]}', 'trainer.logger=[console,wandb]',
        f'trainer.default_local_dir={checkpoints}',
        f'+trainer.rollout_log_dir={run / "rollouts"}',
        f'retriever.url=http://127.0.0.1:{config["port"]}/retrieve',
    ]


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Validate local imports/assets without GPU or training')
    parser.add_argument('--run-name', default=time.strftime('run-%Y%m%d-%H%M%S'))
    parser.add_argument('--runs-dir', type=Path, help='Where to create <run-name>/; default: ./runs')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--python', type=Path, help='Explicit training interpreter; default: the one running this script')
    args = parser.parse_args()
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.run_name):
        parser.error('run-name may contain only letters, numbers, hyphens and underscores')
    config = json.loads((ROOT / 'config.json').read_text())
    if args.gpu is not None:
        config['gpu'] = args.gpu
    python = args.python.absolute() if args.python else Path(sys.executable)
    required = [python, SHARED / 'data/train.parquet', SHARED / 'data/test.parquet',
                SHARED / 'data/e5_Flat.index', SHARED / 'data/corpus.jsonl',
                SHARED / 'models/Qwen2.5-1.5B-Instruct/model.safetensors',
                SHARED / 'models/e5-base-v2/model.safetensors']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit('Missing files: ' + ', '.join(missing))
    run = (args.runs_dir.absolute() if args.runs_dir else ROOT / 'runs') / args.run_name
    # Weights stay beside the workspace even when logs are written elsewhere: the
    # actor checkpoint is ~6GB and a log directory is not the place for it.
    checkpoints = ROOT / 'checkpoints' / args.run_name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(config['gpu']),
               PYTHONPATH=str(ROOT / 'Search-R1'), SEARCH_R1_SHORT_GRPO='1',
               TRAIN_PYTHON=str(python), BASE_MODEL=str(SHARED / 'models/Qwen2.5-1.5B-Instruct'),
               TRAIN_DATA_DIR=str(SHARED / 'data'), TEST_DATA_DIR=str(SHARED / 'data'),
               EXPERIMENT_NAME=f'{ROOT.name}-{args.run_name}', HF_HUB_OFFLINE='1',
               HF_HOME=str(ROOT / '.cache/huggingface'), HF_DATASETS_CACHE=str(ROOT / '.cache/datasets'),
               XDG_CACHE_HOME=str(ROOT / '.cache'), TRITON_CACHE_DIR=str(ROOT / '.cache/triton'),
               TORCH_HOME=str(ROOT / '.cache/torch'), NUMBA_CACHE_DIR=str(ROOT / '.cache/numba'),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               WANDB_MODE='offline', WANDB_ERROR_REPORTING='false', WANDB_DIR=str(run / 'wandb'),
               WANDB_CACHE_DIR=str(ROOT / '.cache/wandb'), WANDB_CONFIG_DIR=str(ROOT / '.cache/wandb-config'),
               WANDB_DATA_DIR=str(run / 'wandb-data'), RAY_ADDRESS='local', HYDRA_FULL_ERROR='1',
               SEARCH_R1_RETRIEVAL_TRACE=str(run / 'retrieval.jsonl'),
               NO_PROXY='localhost,127.0.0.1,' + os.environ.get('NO_PROXY', ''))
    command = ['bash', str(ROOT / 'Search-R1/train_grpo.sh'), *overrides(config, run, checkpoints)]
    if args.check:
        code = ('import pathlib, verl, search_r1, pyarrow.parquet as pq; '
                'print("verl:", verl.__file__); print("search_r1:", search_r1.__file__); '
                f'assert pathlib.Path(verl.__file__).is_relative_to({str(ROOT / "Search-R1")!r}); '
                f'assert all(pathlib.Path(p).is_relative_to({str(ROOT / "Search-R1")!r}) for p in search_r1.__path__); '
                f'assert pq.ParquetFile({str(SHARED / "data/train.parquet")!r}).metadata.num_rows == 1024; '
                f'assert pq.ParquetFile({str(SHARED / "data/test.parquet")!r}).metadata.num_rows == 128')
        subprocess.run([str(python), '-c', code], env=env, cwd=ROOT / 'Search-R1', check=True)
        print(json.dumps({'config': config, 'command': command, 'check': 'passed'}, indent=2))
        return
    import shutil
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise SystemExit('At least 10 GiB free disk space is required')
    used = int(subprocess.check_output(['nvidia-smi', f'--id={config["gpu"]}',
               '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if used >= 100:
        raise SystemExit(f'GPU {config["gpu"]} already uses {used} MiB; refusing to start')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', config['port']))
    run.mkdir(parents=True, exist_ok=False)
    for key in ('WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (ROOT / 'Search-R1').rglob('*.py')
                     if '.venv' not in p.parts}
    (run / 'run_config.json').write_text(json.dumps({'config': config, 'command': command,
                'source_sha256': source_hashes, 'wandb_mode': 'offline'}, indent=2) + '\n')
    retriever = trainer = None
    code = 1
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (run / 'retrieval_server.log').open('w') as log:
            retriever = subprocess.Popen([str(python), str(ROOT / 'serve_retriever.py'),
                '--port', str(config['port'])], cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(120):
            if retriever.poll() is not None:
                raise RuntimeError('Retriever exited; see retrieval_server.log')
            try:
                with opener.open(f'http://127.0.0.1:{config["port"]}/openapi.json', timeout=2):
                    break
            except OSError:
                time.sleep(2)
        else:
            raise RuntimeError('Retriever readiness timeout')
        with (run / 'train.log').open('w') as log:
            trainer = subprocess.Popen(command, cwd=ROOT / 'Search-R1', env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                start_new_session=True)
            for line in trainer.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = trainer.wait()
    finally:
        stop(trainer)
        stop(retriever)
        (run / 'exit_status.json').write_text(json.dumps({'exit_code': code}) + '\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
