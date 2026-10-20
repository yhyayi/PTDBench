"""Private Ray clusters may reuse job IDs on the same physical node."""

from types import SimpleNamespace

import ray

from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import weight_transfer_job_id


def test_socket_namespace_is_unique_between_clusters_and_short(monkeypatch):
    def identifier(node, job):
        monkeypatch.setattr(
            ray, "get_runtime_context",
            lambda: SimpleNamespace(get_node_id=lambda: node, get_job_id=lambda: job),
        )
        return weight_transfer_job_id()

    a = identifier("a" * 56, "01000000")
    assert a == identifier("a" * 56, "01000000")
    assert a != identifier("b" * 56, "01000000")
    assert a != identifier("a" * 56, "02000000")
    assert len(f"/tmp/rl-colocate-zmq-{a}-replica-1234-rank-1234.sock".encode()) < 108
