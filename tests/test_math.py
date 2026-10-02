"""The closed-form write, key statistics, factor controls, and self-query plumbing."""

from __future__ import annotations

import pytest
import torch

from ctw.memory import DenseDelta, LowRankDelta, MemoryState
from ctw.queries import _common_suffix, cloze_questions, parse_questions
from ctw.solve import covariance_ridge, objective
from ctw.stats import KeyStats


def _problem(d_in=24, d_out=10, n=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(d_in, 400, generator=g, dtype=torch.float64)
    x *= torch.linspace(0.1, 3.0, d_in, dtype=torch.float64)[:, None]
    moment = x @ x.T / 400
    keys = torch.randn(d_in, n, generator=g, dtype=torch.float64)
    values = torch.randn(d_out, n, generator=g, dtype=torch.float64)
    return moment, KeyStats.from_moment(moment, 400, 0.0), keys, values


def test_full_rank_solution_is_the_closed_form():
    moment, stats, k, v = _problem()
    ridge = 0.05
    a, b, report = covariance_ridge(k, v, stats, ridge, rank=k.shape[1])
    direct = v @ k.T @ torch.linalg.inv(k @ k.T + ridge * k.shape[0] * moment)
    assert torch.allclose(a.double() @ b.double(), direct, atol=1e-4)
    assert report["rank"] == k.shape[1]


def test_no_crosstalk_without_ridge():
    _, stats, k, v = _problem()
    a, b, report = covariance_ridge(k, v, stats, 1e-9, rank=k.shape[1])
    assert torch.allclose(a.double() @ b.double() @ k, v, atol=1e-4)
    assert report["fit"] > 0.999999


def test_truncation_is_optimal_in_the_hessian_metric():
    moment, stats, k, v = _problem()
    ridge, r = 0.05, 3
    a, b, _ = covariance_ridge(k, v, stats, ridge, rank=r)
    lam = ridge * k.shape[0]
    hessian = k @ k.T + lam * moment
    vals, vecs = torch.linalg.eigh(hessian)
    root, inv_root = (vecs * vals.sqrt()) @ vecs.T, (vecs / vals.sqrt()) @ vecs.T
    full = v @ k.T @ torch.linalg.inv(hessian)
    u, s, vt = torch.linalg.svd(full @ root)
    reference = (u[:, :r] * s[:r]) @ vt[:r] @ inv_root
    assert torch.allclose(a.double() @ b.double(), reference, atol=1e-4)
    best = objective(a, b, k, v, moment, ridge)
    pu, ps, pvt = torch.linalg.svd(full)
    assert best < objective(pu[:, :r] * ps[:r], pvt[:r], k, v, moment, ridge)
    g = torch.Generator().manual_seed(1)
    for _ in range(20):
        da = 1e-2 * torch.randn(a.shape, generator=g)
        db = 1e-2 * torch.randn(b.shape, generator=g)
        assert best <= objective(a + da, b + db, k, v, moment, ridge) + 1e-9


def test_read_key_outside_the_written_keys_reads_nothing():
    """ΔW* z depends on z only through Kᵀ Σ⁻¹ z."""
    moment, stats, k, v = _problem()
    a, b, _ = covariance_ridge(k, v, stats, 0.05, rank=k.shape[1])
    z = torch.randn(k.shape[0], dtype=torch.float64)
    ck = torch.linalg.solve(moment, k)
    z = z - k @ torch.linalg.solve(k.T @ ck, ck.T @ z)   # now kᵢᵀ Σ⁻¹ z = 0 for every key
    assert torch.allclose(k.T @ torch.linalg.solve(moment, z), torch.zeros(k.shape[1], dtype=torch.float64),
                          atol=1e-8)
    assert float((a.double() @ (b.double() @ z)).norm()) < 1e-4 * float(z.norm())


def test_key_stats_inverse_energy_and_shrinkage():
    moment, stats, k, _ = _problem()
    assert torch.allclose(stats.inverse(k), torch.linalg.solve(moment, k), atol=1e-4)
    a, b = torch.randn(10, 2, dtype=torch.float64), torch.randn(2, 24, dtype=torch.float64)
    expected = float(torch.trace(a @ b @ moment @ (a @ b).T))
    assert stats.energy(LowRankDelta(a.float(), b.float())) == pytest.approx(expected, rel=1e-4)
    assert stats.energy(DenseDelta((a @ b).float())) == pytest.approx(expected, rel=1e-4)
    iso = KeyStats(stats.eigvecs, stats.eigvals, 400, 1.0)
    assert torch.allclose(iso.spectrum, torch.full_like(iso.spectrum, float(torch.trace(moment)) / 24))
    assert iso.effective_dim() == pytest.approx(24.0)
    assert 1.0 < stats.effective_dim() < 24.0
    assert stats.whitened_dim() == pytest.approx(24.0, rel=1e-6)
    shrunk = KeyStats(stats.eigvecs, stats.eigvals, 400, 0.3)
    metric = (shrunk.eigvecs.double() * shrunk.spectrum) @ shrunk.eigvecs.double().T
    assert shrunk.whitened_dim() == pytest.approx(float(torch.trace(torch.linalg.solve(metric, moment))), rel=1e-4)
    assert shrunk.whitened_dim() < 24.0


def test_factor_controls_keep_one_factor_and_the_norm():
    s = MemoryState({2: LowRankDelta(torch.randn(8, 3), torch.randn(3, 16))})
    for keep, kept, drawn in (("a", "a", "b"), ("b", "b", "a")):
        r = s.random_like(5, keep=keep)
        d = r.deltas[2]
        assert torch.equal(getattr(d, kept), getattr(s.deltas[2], kept))
        assert not torch.allclose(getattr(d, drawn), getattr(s.deltas[2], drawn))
        assert d.frobenius() == pytest.approx(s.deltas[2].frobenius(), rel=1e-4)
    with pytest.raises(ValueError, match="low-rank"):
        MemoryState({0: DenseDelta(torch.randn(4, 4))}).random_like(0, keep="a")


def test_question_parsing_and_cloze():
    reply = ("1. What was the name of the cat?\n- Who kept the lighthouse?\nQ3: Which year?\n"
             "Not a question.\n2) What was the name of the cat?\n* In which museum is the ledger kept?")
    assert parse_questions(reply, 10) == [
        "What was the name of the cat?", "Who kept the lighthouse?", "In which museum is the ledger kept?"]
    assert parse_questions(reply, 1) == ["What was the name of the cat?"]
    assert cloze_questions("The archive key was wolf. Short one. The emergency key was amber!") == [
        'Complete this sentence from the record: "The archive key was ..."',
        'Complete this sentence from the record: "The emergency key was ..."']
    assert _common_suffix([1, 2, 3, 4], [9, 3, 4]) == 2 and _common_suffix([1], [2]) == 0
