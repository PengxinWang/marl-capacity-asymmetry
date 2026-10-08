"""MAPPO with per-agent (non-shared) actors, centralized critic. Only actors count as 'inference capacity'.

Usage: python mappo.py --env spread --widths 64,64,64 --seed 0 --iters 1500 --out runs/x.pt
"""
import argparse, json, time
import torch, torch.nn as nn
from envs import ENVS


class Actor(nn.Module):
    def __init__(self, d, a, h):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(d, h), nn.Linear(h, h), nn.Linear(h, a)
        self.register_buffer("m1", torch.ones(h)); self.register_buffer("m2", torch.ones(h))

    def forward(self, x):
        x = torch.tanh(self.l1(x)) * self.m1
        x = torch.tanh(self.l2(x)) * self.m2
        return self.l3(x)


def make_critic(d, h=256):
    return nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))


def rollout(env, actors, greedy=False):
    obs = env.reset(); O, A, LP, R = [], [], [], []
    done = False
    while not done:
        acts, lps = [], []
        for i, act in enumerate(actors):
            logits = act(obs[i])
            dist = torch.distributions.Categorical(logits=logits)
            a = logits.argmax(-1) if greedy else dist.sample()
            acts.append(a); lps.append(dist.log_prob(a))
        O.append(obs); A.append(acts); LP.append(lps)
        obs, r, done = env.step(acts); R.append(r)
    return O, A, LP, torch.stack(R)  # R: T,B


@torch.no_grad()
def evaluate(env, actors, n=4, greedy=False):
    return torch.stack([rollout(env, actors, greedy)[3].sum(0).mean() for _ in range(n)]).mean().item()


def train(a):
    torch.manual_seed(a.seed)
    dev = "cuda"
    env = ENVS[a.env](a.B, dev)
    widths = [int(w) for w in a.widths.split(",")]
    assert len(widths) == env.n_agents
    actors = [Actor(env.obs_dims[i], env.act_dims[i], widths[i]).to(dev) for i in range(env.n_agents)]
    critic = make_critic(sum(env.obs_dims)).to(dev)
    params = [p for m in actors + [critic] for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=a.lr)
    log = []
    t0 = time.time()
    for it in range(a.iters):
        with torch.no_grad():
            O, A, LP, R = rollout(env, actors)
            T = len(O)
            S = torch.stack([torch.cat(o, -1) for o in O])  # T,B,D
            V = critic(S).squeeze(-1)
            adv = torch.zeros_like(R); g = 0
            for t in reversed(range(T)):
                nv = V[t + 1] if t + 1 < T else 0
                delta = R[t] + a.gamma * nv - V[t]
                g = delta + a.gamma * a.lam * g; adv[t] = g
            ret = adv + V
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        Obs = [torch.stack([O[t][i] for t in range(T)]).flatten(0, 1) for i in range(env.n_agents)]
        Act = [torch.stack([A[t][i] for t in range(T)]).flatten(0) for i in range(env.n_agents)]
        OLP = [torch.stack([LP[t][i] for t in range(T)]).flatten(0) for i in range(env.n_agents)]
        S, ret, adv = S.flatten(0, 1), ret.flatten(), adv.flatten()
        N = S.shape[0]
        for _ in range(a.epochs):
            perm = torch.randperm(N, device=dev)
            for mb in perm.chunk(a.nmb):
                loss = 0.5 * ((critic(S[mb]).squeeze(-1) - ret[mb]) ** 2).mean()
                for i, act in enumerate(actors):
                    dist = torch.distributions.Categorical(logits=act(Obs[i][mb]))
                    ratio = (dist.log_prob(Act[i][mb]) - OLP[i][mb]).exp()
                    pg = -torch.min(ratio * adv[mb], ratio.clamp(0.8, 1.2) * adv[mb]).mean()
                    loss = loss + pg - a.ent * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 0.5); opt.step()
        if it % 100 == 0 or it == a.iters - 1:
            ep = R.sum(0).mean().item()
            log.append((it, ep)); print(f"it {it} ret {ep:.3f} {time.time()-t0:.0f}s", flush=True)
    final = evaluate(env, actors, n=8)
    torch.save({"args": vars(a), "actors": [x.state_dict() for x in actors], "log": log, "final": final}, a.out)
    print(json.dumps({"final": final}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="spread"); p.add_argument("--widths", default="64,64,64")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--B", type=int, default=512); p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--gamma", type=float, default=0.99); p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=4); p.add_argument("--nmb", type=int, default=4)
    p.add_argument("--ent", type=float, default=0.01); p.add_argument("--out", default="runs/tmp.pt")
    train(p.parse_args())
