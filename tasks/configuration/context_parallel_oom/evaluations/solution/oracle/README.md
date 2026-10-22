# Canonical solution oracle

## Root cause

The model code is healthy (`transformer_impl.py` already applies
`torch.roll(..., shifts=-1)`, and `taco.py` is the healthy scorer). The
launch script's context parallel is 1.

- `examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh`
  - kind: `config`
  - observed: `ULYSSES_SEQUENCE_PARALLEL_SIZE` defaults to 1 and is passed as
    `actor_rollout_ref.actor.ulysses_sequence_parallel_size` and
    `actor_rollout_ref.ref.ulysses_sequence_parallel_size`
  - expected: both set to 4
  - note: on this FSDP recipe that field is cp. cp=8 does not divide
    Qwen2.5-1.5B's 12 attention heads.

## Scoring

Machine grading reads only the new experiment logs. It does not grade TACO
accuracy. Pass requires every condition below; otherwise `score=0`:

- every resolved `max_response_length` is 30720
- the log has no CUDA OOM
- an actor update is present (`step:` together with an actor or reward metric)
