"""Per-agent diagnostics on Leader env: final distance to goal of each agent, and the same when the OTHER agents
are frozen (always action 0 = no-op). If a narrow agent is a 'follower', freezing its wide teammate should hurt it
far more than it hurts a solo agent of the same width (which must compute f(z) itself)."""
import sys, glob, torch
from probe import load


@torch.no_grad()
def per_agent_dist(env, actors, freeze=()):
    obs = env.reset(); done = False
    while not done:
        acts = []
        for i, a in enumerate(actors):
            x = a(obs[i]).argmax(-1) if True else None
            acts.append(torch.zeros_like(x) if i in freeze else torch.distributions.Categorical(logits=a(obs[i])).sample())
        obs, r, done = env.step(acts)
    return (env.pos - env.goal[:, None]).norm(dim=-1).mean(0).tolist()


torch.manual_seed(0)
for p in sorted(glob.glob(sys.argv[1])):
    env, actors, a = load(p)
    n = len(actors)
    line = f"{a['env']:8s} {a['widths']:9s} s{a['seed']}  dist={[round(x,3) for x in per_agent_dist(env, actors)]}"
    if n > 1:
        for i in range(n):
            d = per_agent_dist(env, actors, freeze=[j for j in range(n) if j != i])
            line += f"  others-frozen a{i}={d[i]:.3f}"
    print(line)
