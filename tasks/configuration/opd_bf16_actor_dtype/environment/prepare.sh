#!/usr/bin/env bash
set -euo pipefail

# Python environment (uv, see environment/setup_uv.sh). It is built here because prepare is the only
# stage with network access; the interpreter and the venv both live under $PTDBENCH_PREPARED, which the
# agent and verifier stages see at the same path.
export UV_PYTHON_INSTALL_DIR="$PTDBENCH_PREPARED/uv-python"
bash "$PTDBENCH_ENVIRONMENT/setup_uv.sh" "$PTDBENCH_PREPARED/venv"
{
    echo "VIRTUAL_ENV=$PTDBENCH_PREPARED/venv"
    echo "PATH=$PTDBENCH_PREPARED/venv/bin:$PATH"
} >> "$PTDBENCH_PREPARED/env"

# Student and teacher checkpoints (immutable Hugging Face revisions).
ptdbench_fetch_hf \
    --repo deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --revision 73d63fa6f97d4285b49bdbedf6a161ceba1e55ad \
    --allow config.json --allow generation_config.json \
    --allow tokenizer.json --allow tokenizer_config.json \
    --allow model.safetensors \
    --expect model.safetensors=58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945 \
    --expect-size model.safetensors=3554214621 \
    --dest "$PTDBENCH_WORKSPACE/model/DeepSeek-R1-Distill-Qwen-1.5B"

ptdbench_fetch_hf \
    --repo hbx/JustRL-DeepSeek-1.5B \
    --revision 0637e4096c789c67f9eecbe8355e0bdeddede1c2 \
    --allow config.json --allow generation_config.json \
    --allow tokenizer.json --allow tokenizer_config.json \
    --allow model.safetensors \
    --expect model.safetensors=1bedcdf243c4e2e633fa04c05617809cf6a1bbc1a07221609035f6347efecffb \
    --expect-size model.safetensors=3554214752 \
    --dest "$PTDBENCH_WORKSPACE/model/JustRL-DeepSeek-1.5B"

# verl-format DAPO-Math-17k training set and the validation sets referenced by the training script,
# pinned to a commit of the public thunlp/OPD repository.
OPD_RAW=https://raw.githubusercontent.com/thunlp/OPD/ac26e38d6f1572eb027597b48a9f4e01f6915ef8

ptdbench_fetch_url \
    --url "$OPD_RAW/datasets/dapo-math-17k.parquet" \
    --expect 039f3afd689c846985bd2bf58e55a2210a8b08a1a9e1f60dbc07107a5f341925 \
    --expect-size 3613818 \
    --dest "$PTDBENCH_WORKSPACE/datasets/dapo-math-17k.parquet"

ptdbench_fetch_url \
    --url "$OPD_RAW/datasets/test_data/AIME24/test.parquet" \
    --expect dc3ae15e0236828eb95033a68aab60e19da7ea1afdaad80e26924e5f82d3d96f \
    --expect-size 15217 \
    --dest "$PTDBENCH_WORKSPACE/datasets/test_data/AIME24/test.parquet"

ptdbench_fetch_url \
    --url "$OPD_RAW/datasets/test_data/AIME25/test.parquet" \
    --expect 14ee9a0e8d79cba76c62247bc5ebee1647e023eccf60e31994e279113777dad2 \
    --expect-size 13112 \
    --dest "$PTDBENCH_WORKSPACE/datasets/test_data/AIME25/test.parquet"

ptdbench_fetch_url \
    --url "$OPD_RAW/datasets/test_data/AMC23/test.parquet" \
    --expect 0f933978194b73fc091bd667af7d9056d3d745c9d9e3577b5adcad77f8f1ae7c \
    --expect-size 21835 \
    --dest "$PTDBENCH_WORKSPACE/datasets/test_data/AMC23/test.parquet"
