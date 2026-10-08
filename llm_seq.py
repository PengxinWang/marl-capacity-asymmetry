"""LLM probe: 2-agent sequential collaboration on MATH-500 with IDENTICAL prompts (asymmetry only from position).
Agent A writes first; agent B sees A's message and gives the team's final answer. Swap each agent's checkpoint
size independently and measure team accuracy. Stage outputs are cached per model so the grid is cheap.

Usage: python llm_seq.py stage1 <model>      -> drafts/<model>.jsonl
       python llm_seq.py stage2 <modelA> <modelB>
"""
import sys, json, os, importlib.util
from vllm import LLM, SamplingParams

spec = importlib.util.spec_from_file_location("g", "/localwork/pxwang24/rl_lab/RobustLLMAgent/verl/utils/reward_score/math.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

HUB = os.path.expanduser("~/.cache/huggingface/hub")
DATA = [json.loads(l) for l in open([os.path.join(r, f) for r, _, fs in os.walk(HUB + "/datasets--HuggingFaceH4--MATH-500") for f in fs if f == "test.jsonl"][0])]
SYS = ("You are one member of a two-person team solving a math problem. You will see the problem and the team "
       "discussion so far (possibly empty). Write your contribution to the discussion, reasoning step by step, "
       "and end with the final answer in \\boxed{}.")


def prompt(tok, prob, disc):
    d = disc if disc else "(empty)"
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Problem:\n{prob}\n\nTeam discussion so far:\n{d}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def score(text, ans):
    b = g.last_boxed_only_string(text)
    try:
        return float(b is not None and g.is_equiv(g.remove_boxed(b), ans))
    except Exception:
        return 0.0


def llm(m):
    return LLM(f"Qwen/Qwen2.5-{m}-Instruct", gpu_memory_utilization=0.3, max_model_len=4096, seed=0)


sp = SamplingParams(temperature=0.0, max_tokens=1024)
os.makedirs("llm_out", exist_ok=True)
if sys.argv[1] == "stage1":
    m = sys.argv[2]; L = llm(m); tok = L.get_tokenizer()
    outs = L.generate([prompt(tok, d["problem"], "") for d in DATA], sp)
    with open(f"llm_out/A_{m}.jsonl", "w") as f:
        for d, o in zip(DATA, outs):
            t = o.outputs[0].text
            f.write(json.dumps({"a": t, "acc": score(t, d["answer"])}) + "\n")
else:
    mB = sys.argv[2]; L = llm(mB); tok = L.get_tokenizer()
    for mA in sys.argv[3].split(","):
        drafts = [json.loads(l) for l in open(f"llm_out/A_{mA}.jsonl")]
        outs = L.generate([prompt(tok, d["problem"], "Teammate 1: " + x["a"]) for d, x in zip(DATA, drafts)], sp)
        acc = [score(o.outputs[0].text, d["answer"]) for d, o in zip(DATA, outs)]
        r = {"A": mA, "B": mB, "team_acc": sum(acc) / len(acc), "A_alone": sum(x["acc"] for x in drafts) / len(drafts)}
        print("RESULT", json.dumps(r), flush=True)
        open("llm_out/grid.jsonl", "a").write(json.dumps(r) + "\n")
