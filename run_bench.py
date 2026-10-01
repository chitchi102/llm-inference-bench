from bench import prompts,measure,table,engines
from transformers import AutoTokenizer
import argparse,json
from pathlib import Path

def run(gen, engine, batch_sizes, prompt_len, output_len, vocab_size, warmup, repeat):
    res = []
    for b in batch_sizes:
        pmp = prompts.make_prompts(batch_size=b,prompt_len=prompt_len,vocab_size=vocab_size)
        generated = gen(pmp,output_len)
        if not (len(generated) == b and all(len(o) == output_len for o in generated)):
            raise RuntimeError("output's size is not correct")
        g1 = lambda :gen(pmp,1)
        go = lambda :gen(pmp,output_len)
        ttft = measure.time_calls(g1,warmup=warmup,repeat=repeat)
        total = measure.time_calls(go,warmup=warmup,repeat=repeat)
        summ = measure.summarize(ttft_s=ttft,total_s=total,batch_size=b,output_len=output_len)
        info = {"engine":engine,"batch_size":b,"prompt_len":prompt_len,"output_len":output_len}
        res.append({**summ,**info})
    return res



def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--engine",required=True)
    p.add_argument("--model",default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--device",default="cuda")
    p.add_argument("--batch-sizes",type=int,nargs="+",default=[1,8,32])
    p.add_argument("--prompt-len",type=int,default=128)
    p.add_argument("--output-len",type=int,default=128)
    p.add_argument("--warmup",type=int,default=1)
    p.add_argument("--repeat",type=int,default=3)
    p.add_argument("--out",required=True)
    args = p.parse_args(argv)
    tok = AutoTokenizer.from_pretrained(args.model)
    vocab_size = len(tok)
    gen = engines.make_engine(args.engine,args.model,args.device)
    res = run(gen, args.engine, args.batch_sizes, args.prompt_len, args.output_len, vocab_size, args.warmup, args.repeat)
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(res,indent=2))
    print(table.to_markdown(res))
    
if __name__=="__main__":
    main()