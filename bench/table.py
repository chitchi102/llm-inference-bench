def to_markdown(rows):
    res = ["| engine | batch | TTFT (ms) | TPOT (ms) | throughput (tok/s) |","|---|---:|---:|---:|---:|"]
    for row in sorted(rows,key=lambda r: (r["batch_size"],r["engine"])):
        res.append(f"| {row['engine']} | {row['batch_size']} | {row['ttft_ms']:.1f} | {row['tpot_ms']:.2f} | {row['throughput_tok_s']:.0f} |")
    return "\n".join(res)