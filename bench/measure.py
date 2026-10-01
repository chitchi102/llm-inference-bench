import time,statistics

def time_calls(fn,warmup,repeat):
    if repeat < 1:
        raise ValueError("repeat is smaller than 1.")
    for _ in range(warmup):
        fn()
    time_list = []
    for _ in range(repeat):
        bef_time = time.perf_counter()
        fn()
        aft_time = time.perf_counter()
        time_list.append(aft_time-bef_time)
    return time_list

def summarize(ttft_s, total_s, batch_size, output_len):
    if output_len < 2:
        raise ValueError("output_len is smaller than 2.")
    med_ttft_s = statistics.median(ttft_s)
    med_total_s = statistics.median(total_s)
    ttft_ms = med_ttft_s * 1000
    tpot_ms = ((med_total_s-med_ttft_s)/(output_len-1)) * 1000
    throughput_tok_s = batch_size*output_len/med_total_s
    return {"ttft_ms":ttft_ms,"tpot_ms":tpot_ms,"throughput_tok_s":throughput_tok_s}



