"""1-b: HF で生成 → 測る → json と表（手元の CPU・超小型のランダム Llama で通す）。"""
import json

import pytest

from bench.engines import hf_generate, load_hf
from bench.table import to_markdown
from run_bench import main, run

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def tiny():
    return load_hf(TINY, "cpu")


# ---- hf_generate ----

def test_hf_returns_one_list_per_prompt(tiny):
    out = hf_generate(tiny, [[1, 2, 3, 4], [5, 6, 7, 8]], new_tokens=5)
    assert isinstance(out, list)
    assert len(out) == 2


def test_hf_returns_only_new_tokens(tiny):
    out = hf_generate(tiny, [[1, 2, 3, 4], [5, 6, 7, 8]], new_tokens=5)
    assert all(len(o) == 5 for o in out)          # プロンプトの 4 個は含めない
    assert all(isinstance(t, int) for o in out for t in o)


def test_hf_is_deterministic(tiny):
    prompts = [[10, 20, 30]]
    assert hf_generate(tiny, prompts, 6) == hf_generate(tiny, prompts, 6)


def test_hf_does_not_stop_early_at_eos(tiny):
    # 最初に出るトークンをわざと「終わり」の印にしても、指定した数だけ出し続けること
    prompts = [[10, 20, 30]]
    first = hf_generate(tiny, prompts, 1)[0][0]
    old = tiny.generation_config.eos_token_id
    tiny.generation_config.eos_token_id = first
    try:
        out = hf_generate(tiny, prompts, 6)
    finally:
        tiny.generation_config.eos_token_id = old
    assert len(out[0]) == 6


# ---- run（生成関数は偽物でよい） ----

def fake_gen(prompts, n):
    return [[0] * n for _ in prompts]


def test_run_one_row_per_batch_size():
    rows = run(fake_gen, "fake", [1, 2, 4], prompt_len=8, output_len=4,
               vocab_size=100, warmup=0, repeat=1)
    assert [r["batch_size"] for r in rows] == [1, 2, 4]
    assert all(r["engine"] == "fake" for r in rows)
    for r in rows:
        assert {"ttft_ms", "tpot_ms", "throughput_tok_s", "prompt_len", "output_len"} <= set(r)


def test_run_passes_right_shaped_prompts():
    seen = []

    def spy(prompts, n):
        seen.append((len(prompts), len(prompts[0]), n))
        return fake_gen(prompts, n)

    run(spy, "spy", [3], prompt_len=8, output_len=4, vocab_size=100, warmup=0, repeat=1)
    assert (3, 8, 1) in seen          # TTFT 用＝1 トークン
    assert (3, 8, 4) in seen          # 全体用＝output_len トークン


def test_run_rejects_short_outputs():
    # 途中で止まる生成を測ると速く見えてしまう＝先に気づいて止めること
    def short_gen(prompts, n):
        return [[0] * max(1, n - 1) for _ in prompts]

    with pytest.raises(RuntimeError):
        run(short_gen, "short", [2], prompt_len=8, output_len=4,
            vocab_size=100, warmup=0, repeat=1)


# ---- 表 ----

def test_table_sorted_by_batch_then_engine():
    rows = [
        {"engine": "vllm", "batch_size": 8, "ttft_ms": 1, "tpot_ms": 1, "throughput_tok_s": 1},
        {"engine": "hf", "batch_size": 8, "ttft_ms": 2, "tpot_ms": 2, "throughput_tok_s": 2},
        {"engine": "hf", "batch_size": 1, "ttft_ms": 3, "tpot_ms": 3, "throughput_tok_s": 3},
    ]
    lines = to_markdown(rows).splitlines()
    assert lines[0].startswith("| engine")
    assert [line.split("|")[1].strip() + line.split("|")[2].strip() for line in lines[2:]] == [
        "hf1", "hf8", "vllm8",
    ]


def test_table_number_format():
    row = {"engine": "hf", "batch_size": 1, "ttft_ms": 12.345, "tpot_ms": 6.789,
           "throughput_tok_s": 150.4}
    assert to_markdown([row]).splitlines()[2] == "| hf | 1 | 12.3 | 6.79 | 150 |"


# ---- 端から端まで（コマンド1本で json ができる） ----

def test_main_end_to_end(tmp_path):
    out = tmp_path / "results" / "hf.json"
    main(["--engine", "hf", "--model", TINY, "--device", "cpu",
          "--batch-sizes", "1", "2", "--prompt-len", "8", "--output-len", "4",
          "--warmup", "0", "--repeat", "1", "--out", str(out)])
    rows = json.loads(out.read_text())
    assert [r["batch_size"] for r in rows] == [1, 2]
    assert all(r["throughput_tok_s"] > 0 for r in rows)
