DATA_DIR=/user/chenyinghao/datasets

python examples/search-r1/local_dense_retriever/retrieval_server.py \
    --index_path  $DATA_DIR/wiki-18/e5_Flat.index \
    --corpus_path $DATA_DIR/wiki-18/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu
