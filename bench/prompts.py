import random

def make_prompts(batch_size, prompt_len, vocab_size, seed=0):
    rng = random.Random(seed)
    prompts = []
    for _ in range(batch_size):
        prompt = []
        for _ in range(prompt_len):
            prompt.append(rng.randrange(vocab_size))
        prompts.append(prompt)
    return prompts