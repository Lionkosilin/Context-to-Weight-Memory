from __future__ import annotations

import json

import pytest

from ctw.cli import main
from ctw.run import run
from ctw.writers import WRITERS

from .conftest import config_for


def _check(report, splits):
    assert set(report["evaluations"]) == set(splits)
    for split in splits:
        arms = report["evaluations"][split]
        assert {"off", "context", "write"} <= set(arms)
        for r in arms.values():
            assert r["n"] > 0 and len(r["rows"]) == r["n"]


ACWC = {"writer": {"name": "acwc", "params": {"steps": 6, "eval_every": 3, "key_rank": 4}},
        "task": {"name": "synthetic_kv", "params": {"train_docs": 4, "dev_docs": 2, "test_docs": 2}},
        "memory": {"layers": [-1]}}


@pytest.mark.parametrize("arch", ["qwen3", "llama", "gpt2", "gpt_neox", "opt"])
def test_acwc_runs_on_every_layout(arch, model_dirs, tmp_path):
    cfg = config_for(model_dirs[arch], tmp_path, name=f"acwc-{arch}", **ACWC)
    report = run(cfg)
    _check(report, ["seen_recombination", "unseen_values"])
    ev = report["evaluations"]["seen_recombination"]
    assert {"wrong", "random"} <= set(ev)
    assert report["state_params_per_document"] == 4 * (64 + 32) or arch == "gpt2"
    assert report["fit"]["marks"][-1]["step"] == 6


def test_acwc_save_load_write_ask(model_dirs, tmp_path, capsys):
    cfg = config_for(model_dirs["qwen3"], tmp_path, name="acwc-io", **ACWC)
    cfg["writer"]["save"] = str(tmp_path / "compiler.pt")
    run(cfg)
    doc = tmp_path / "doc.txt"
    doc.write_text("The archive key was wolf. The emergency key was amber.")
    cfgfile = tmp_path / "cfg.yaml"
    import yaml
    cfgfile.write_text(yaml.safe_dump(cfg))
    state = tmp_path / "doc.safetensors"
    main(["write", str(cfgfile), "--document", str(doc), "--out", str(state),
          "--set", f"writer.load={tmp_path / 'compiler.pt'}"])
    main(["ask", str(cfgfile), "--state", str(state), "--question", "What was the archive key?",
          "--max-new-tokens", "2"])
    out = capsys.readouterr().out
    assert "[with memory]" in out and "params=" in out


CCD = {"rank": 6, "steps": 3, "questions": 2, "answer_tokens": 3}


def test_ccd_synthetic_with_controls_and_selectivity(model_dirs, tmp_path):
    cfg = config_for(model_dirs["qwen3"], tmp_path, name="ccd",
                     writer={"name": "ccd", "params": CCD},
                     task={"name": "synthetic_kv", "params": {"test_docs": 2}},
                     memory={"layers": [1, 3]}, eval={"selectivity": True})
    report = run(cfg)
    _check(report, ["seen_recombination", "unseen_values"])
    ev = report["evaluations"]["seen_recombination"]
    assert {"wrong", "random", "random_keys", "random_values"} <= set(ev)
    assert all("mean_log10_selectivity" in ev[a] for a in ("write", "random", "random_keys"))
    assert report["state_params_per_document"] == 2 * 6 * (64 + 32)
    stats_files = list((tmp_path / "stats").rglob("*.pt"))
    assert len(stats_files) == 2


def test_ccd_write_reports_and_reaches_its_targets(model_dirs, tmp_path):
    from ctw.run import setup
    from ctw.writers import build_writer

    cfg = config_for(model_dirs["llama"], tmp_path, memory={"layers": [0, 2]})
    ctx = setup(cfg)
    doc = "The archive key was wolf. The emergency key was amber. The navigation key was silver."
    state = build_writer("ccd", {**CCD, "steps": 20, "rank": 64, "ridge": 1e-4}).write(ctx, doc)
    meta = state.meta
    assert meta["cloze"] == 3 and meta["queries"] >= 3
    assert meta["kl_target"] <= meta["kl_residual"] + 1e-6
    assert set(meta["layers"]) == {0, 2}
    assert all(r["fit"] > 0.9 and r["disturbance"] >= 0 for r in meta["layers"].values())
    assert state.lowrank and state.frobenius() > 0


def test_ccd_memo_with_scales(model_dirs, tmp_path):
    cfg = config_for(model_dirs["gpt2"], tmp_path, name="ccd-memo",
                     writer={"name": "ccd", "params": CCD},
                     task={"name": "memo"}, memory={"layers": [1, 2]}, eval={"scales": [0.5, 1.0]})
    report = run(cfg)
    ev = report["evaluations"]["memo"]
    assert {"write@0.5", "write@1", "wrong", "random", "random_values"} <= set(ev)
    assert ev["write@1"]["n"] == 6


@pytest.mark.parametrize("writer,params", [
    ("ntp", {"passes": 1, "chunk": 64}),
    ("dcd", {"passes": 1, "rank": 4, "max_norm": 1.0, "questions": 2, "answer_tokens": 2}),
])
def test_gradient_writers_on_json_bank(writer, params, model_dirs, tmp_path):
    cfg = config_for(model_dirs["llama"], tmp_path, name=writer,
                     writer={"name": writer, "params": params},
                     task={"name": "qa_json", "params": {"path": "examples/lighthouse.json", "split": "check"}},
                     memory={"layers": "last:2"},
                     eval={"export_txt": True, "heldout_text": "LICENSE"})
    report = run(cfg)
    _check(report, ["qa"])
    ev = report["evaluations"]["qa"]
    assert ev["write"]["n"] == 3 and "mean_heldout_ppl" in ev["write"]
    assert ("random_values" in ev) == (writer == "dcd")
    exported = list((tmp_path / writer / "seed7" / "qa" / "write").glob("*.txt"))
    assert len(exported) == 3 and "[REFERENCE]" in exported[0].read_text()


def test_dcd_learns_from_a_document_shorter_than_one_chunk(model_dirs, tmp_path):
    """Teacher and student inputs differ for every query, so the loss and ΔW are nonzero."""
    from ctw.queries import synthesize
    from ctw.run import setup
    from ctw.writers import build_writer

    ctx = setup(config_for(model_dirs["qwen3"], tmp_path, memory={"layers": "last:2"}))
    doc = json.loads(open("examples/lighthouse.json").read())["episodes"][0]["document"]
    for q in synthesize(ctx, doc, 2, True, 2):
        assert q.teacher.shape[1] > q.student.shape[1]
    state = build_writer("dcd", {"passes": 1, "questions": 2, "answer_tokens": 2}).write(ctx, doc)
    assert state.meta["steps"] > 0 and state.frobenius() > 1e-3


def test_summarize_two_seeds(model_dirs, tmp_path, capsys):
    paths = []
    for seed in (7, 11):
        cfg = config_for(model_dirs["qwen3"], tmp_path, name="sum", seed=seed, **ACWC)
        run(cfg)
        paths.append(str(tmp_path / "sum" / f"seed{seed}.json"))
    main(["summarize", *paths, "--out", str(tmp_path / "agg.json")])
    agg = json.loads((tmp_path / "agg.json").read_text())
    assert agg["seeds"] == [7, 11]
    arm = agg["splits"]["seen_recombination"]["arms"]["write"]
    assert arm["n"] == 16 and len(arm["per_seed"]) == 2


def test_inspect_and_list(model_dirs, capsys):
    main(["inspect", "--model", model_dirs["opt"], "--device", "cpu", "--layers", "last:2", "--rank", "4"])
    info = json.loads(capsys.readouterr().out)
    assert info["out_proj"] == "fc2" and info["selected_layers"] == [0, 1]
    main(["list"])
    listed = capsys.readouterr().out
    assert all(f"writer {name}" in listed for name in WRITERS)


def test_unknown_writer_param_is_rejected():
    from ctw.writers import build_writer
    with pytest.raises(ValueError, match="unknown params"):
        build_writer("acwc", {"stepz": 3})


def test_plugin_writer_from_file(model_dirs, tmp_path):
    plugin = tmp_path / "zero_writer.py"
    plugin.write_text(
        "import torch\n"
        "from ctw.memory import DenseDelta, MemoryState\n"
        "from ctw.writers import Writer, register\n"
        "@register('zero')\n"
        "class Zero(Writer):\n"
        "    def write(self, ctx, document):\n"
        "        return MemoryState({i: DenseDelta(torch.zeros(ctx.adapter.dims(i)[::-1])) for i in ctx.layers})\n")
    cfgfile = tmp_path / "cfg.yaml"
    import yaml
    cfg = config_for(model_dirs["gpt2"], tmp_path, name="zero", imports=[str(plugin)],
                     writer={"name": "zero"}, task={"name": "memo"}, memory={"layers": [0]})
    cfgfile.write_text(yaml.safe_dump(cfg))
    main(["run", str(cfgfile)])
    report = json.loads((tmp_path / "zero" / "seed7.json").read_text())
    ev = report["evaluations"]["memo"]
    assert [r["reply"] for r in ev["write"]["rows"]] == [r["reply"] for r in ev["off"]["rows"]]
