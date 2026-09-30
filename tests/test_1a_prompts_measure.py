"""1-a: GPU もモデルも要らない部品（手元の CPU で通す）。"""
import time

import pytest

from bench.measure import summarize, time_calls
from bench.prompts import make_prompts


# ---- make_prompts ----

def test_prompts_shape():
    prompts = make_prompts(batch_size=4, prompt_len=16, vocab_size=100)
    assert len(prompts) == 4
    assert all(len(p) == 16 for p in prompts)


def test_prompts_are_ints_in_vocab():
    prompts = make_prompts(batch_size=8, prompt_len=64, vocab_size=50)
    for p in prompts:
        for t in p:
            assert isinstance(t, int)
            assert 0 <= t < 50


def test_prompts_same_seed_same_prompts():
    assert make_prompts(3, 10, 1000, seed=1) == make_prompts(3, 10, 1000, seed=1)


def test_prompts_rows_differ():
    # 同じプロンプトを並べると vLLM のキャッシュで速く見えてしまう＝行ごとに違うこと
    prompts = make_prompts(batch_size=4, prompt_len=32, vocab_size=1000)
    assert len({tuple(p) for p in prompts}) == 4


# ---- time_calls ----

def test_time_calls_counts_warmup_and_repeat():
    calls = []
    times = time_calls(lambda: calls.append(1), warmup=2, repeat=3)
    assert len(calls) == 5          # 2 回は捨てる・3 回だけ測る
    assert len(times) == 3


def test_time_calls_measures_seconds():
    times = time_calls(lambda: time.sleep(0.02), warmup=0, repeat=2)
    assert all(0.015 < t < 0.5 for t in times)


def test_time_calls_warmup_not_timed():
    # 1 回目だけ遅い関数：ウォームアップで捨てれば測った値はどれも速い
    state = {"first": True}

    def fn():
        if state["first"]:
            state["first"] = False
            time.sleep(0.2)

    times = time_calls(fn, warmup=1, repeat=3)
    assert max(times) < 0.1


def test_time_calls_rejects_zero_repeat():
    with pytest.raises(ValueError):
        time_calls(lambda: None, warmup=0, repeat=0)


# ---- summarize ----

def test_summarize_known_numbers():
    row = summarize(
        ttft_s=[0.1, 0.2, 0.3],       # 中央値 0.2 秒
        total_s=[1.0, 1.2, 5.0],      # 中央値 1.2 秒（5.0 は外れ値）
        batch_size=4,
        output_len=11,
    )
    assert row["ttft_ms"] == pytest.approx(200.0)
    assert row["tpot_ms"] == pytest.approx(100.0)            # (1.2 - 0.2) / (11 - 1)
    assert row["throughput_tok_s"] == pytest.approx(4 * 11 / 1.2)


def test_summarize_keys():
    row = summarize([0.1], [1.0], batch_size=1, output_len=2)
    assert set(row) == {"ttft_ms", "tpot_ms", "throughput_tok_s"}


def test_summarize_rejects_output_len_1():
    with pytest.raises(ValueError):
        summarize([0.1], [0.1], batch_size=1, output_len=1)


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([], [1.0], batch_size=1, output_len=4)
    with pytest.raises(ValueError):
        summarize([0.1], [], batch_size=1, output_len=4)
