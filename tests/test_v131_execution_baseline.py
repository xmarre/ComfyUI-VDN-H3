from pathlib import Path


def test_diagnostic_branch_restores_v131_stateless_execution_contract():
    source = (Path(__file__).resolve().parents[1] / "vdn_h3" / "nodes.py").read_text()

    assert "from vdn_h3.branch import LinearBranch" in source
    assert "from vdn_h3.retained import RuntimeLinearBranch" not in source
    assert "RuntimeLinearBranch(" not in source
    assert "LinearBranch(" in source
    assert 'branch_weights = "stream"' in source
    assert "retain = False" in source
    assert "fast_kernels = False" in source
    assert "managed_weights=None" in source
    assert "baseline=v1.3.1-sync" in source
