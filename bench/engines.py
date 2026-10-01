from transformers import AutoModelForCausalLM
import torch

def load_hf(model_name, device):
    dtype = torch.bfloat16 if device=="cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name,dtype=dtype)
    return model.to(device).eval()

def hf_generate(model,prompts,new_tokens):
    pmps = torch.tensor(prompts,device=model.device.type)
    with torch.inference_mode():
        outputs = model.generate(pmps,max_new_tokens=new_tokens,min_new_tokens=new_tokens,do_sample=False,attention_mask=torch.ones_like(pmps))
        if model.device.type == "cuda":
            torch.cuda.synchronize()
    return outputs[:,len(pmps[0]):].tolist()

def make_engine(name, model_name, device="cuda"):
    infra_name = ["hf"]
    if name in infra_name:
        if name == "hf":
            model = load_hf(model_name=model_name,device=device)
            gen = lambda prompts,new_tokens: hf_generate(model,prompts,new_tokens)
            return gen
    else:
        raise ValueError("unknown engine" + name)
