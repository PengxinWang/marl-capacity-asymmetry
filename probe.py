"""Per-agent capacity probe: mask a fraction of hidden units in ONE agent's actor (others intact), measure team return.

Normalized drop: dJ = (J_full - J_masked) / (J_full - J_rand), where J_rand = all agents uniformly random.
Two masking rules: 'mag' (keep units with largest outgoing-weight norm, both hidden layers) and 'rand' (random units, avg 3 draws).
"""
import sys, json, glob
import torch
from envs import ENVS
from mappo import Actor, evaluate


class Rand(torch.nn.Module):
    def __init__(self, a): super().__init__(); self.a = a
    def forward(self, x): return torch.zeros(x.shape[0], self.a, device=x.device)


def load(path, B=4096):
    ck = torch.load(path, map_location="cuda")
    a = ck["args"]; env = ENVS[a["env"]](B, "cuda")
    ws = [int(w) for w in a["widths"].split(",")]
    actors = []
    for i, sd in enumerate(ck["actors"]):
        m = Actor(env.obs_dims[i], env.act_dims[i], ws[i]).cuda(); m.load_state_dict(sd); actors.append(m)
    return env, actors, a


def set_mask(m, frac, rule, g=None):
    h = m.m1.numel(); k = max(1, round(h * (1 - frac)))
    for mask, nxt in ((m.m1, m.l2), (m.m2, m.l3)):
        if rule == "mag":
            keep = nxt.weight.norm(dim=0).topk(k).indices
        else:
            keep = torch.randperm(h, generator=g)[:k].cuda()
        mask.zero_(); mask[keep] = 1


@torch.no_grad()
def probe(path, fracs=(0.5, 0.75, 0.875)):
    torch.manual_seed(0)
    env, actors, a = load(path)
    J = evaluate(env, actors, n=4)
    Jr = evaluate(env, [Rand(d) for d in env.act_dims], n=4)
    res = {"path": path, "env": a["env"], "widths": a["widths"], "seed": a["seed"], "J": J, "J_rand": Jr, "drop": {}}
    for i in range(len(actors)):
        for f in fracs:
            for rule in ("mag", "rand"):
                vals = []
                for s in range(1 if rule == "mag" else 3):
                    set_mask(actors[i], f, rule, torch.Generator().manual_seed(s))
                    vals.append(evaluate(env, actors, n=2))
                actors[i].m1.fill_(1); actors[i].m2.fill_(1)
                Jm = sum(vals) / len(vals)
                res["drop"][f"a{i}_{rule}_{f}"] = (J - Jm) / (J - Jr)
    return res


if __name__ == "__main__":
    out = [probe(p) for p in sorted(glob.glob(sys.argv[1]))]
    for r in out:
        print(json.dumps(r))
