"""Serve this workspace's retrieval implementation with local assets."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SHARED = ROOT / 'shared'
sys.path.insert(0, str(ROOT / 'Search-R1'))
import faiss
import uvicorn
from search_r1.search import retrieval_server as server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    faiss.omp_set_num_threads(1)
    server.config = server.Config(
        retrieval_method='e5', index_path=str(SHARED / 'data/e5_Flat.index'),
        corpus_path=str(SHARED / 'data/corpus.jsonl'), retrieval_topk=3, faiss_gpu=False,
        retrieval_model_path=str(SHARED / 'models/e5-base-v2'), retrieval_pooling_method='mean',
        retrieval_query_max_length=256, retrieval_use_fp16=True, retrieval_batch_size=64,
    )
    server.retriever = server.get_retriever(server.config)
    uvicorn.run(server.app, host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
