"""Vectorized MPE-style cooperative envs in torch (all envs step in parallel on GPU).

Each env exposes: n_agents, obs_dims[i], act_dims[i], reset() -> list[obs_i], step(list[act_i]) -> (obs, team_reward, done).
All agents get the same team reward. Actions are discrete.
"""
import torch

MOVES = torch.tensor([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.float32)


class Base:
    T = 25

    def __init__(self, B, device):
        self.B, self.dev = B, device
        self.moves = MOVES.to(device)

    def _move(self, pos, vel, a):
        vel = 0.75 * vel + 0.1 * self.moves[a]
        return (pos + vel).clamp(-1.5, 1.5), vel


class Spread(Base):
    """Symmetric cooperative navigation: N agents cover N landmarks. Identical obs/act structure for every agent."""

    def __init__(self, B, device, N=3):
        super().__init__(B, device)
        self.N = self.n_agents = N
        d = 4 + 2 * N + 2 * (N - 1)
        self.obs_dims, self.act_dims = [d] * N, [5] * N

    def reset(self):
        B, N, dev = self.B, self.N, self.dev
        self.pos = torch.rand(B, N, 2, device=dev) * 2 - 1
        self.vel = torch.zeros(B, N, 2, device=dev)
        self.lm = torch.rand(B, N, 2, device=dev) * 2 - 1
        self.t = 0
        return self._obs()

    def _obs(self):
        out = []
        for i in range(self.N):
            p = self.pos[:, i]
            others = torch.cat([self.pos[:, j] - p for j in range(self.N) if j != i], -1)
            out.append(torch.cat([self.vel[:, i], p, (self.lm - p[:, None]).flatten(1), others], -1))
        return out

    def step(self, acts):
        a = torch.stack(acts, 1)
        self.pos, self.vel = self._move(self.pos, self.vel, a)
        d = torch.cdist(self.lm, self.pos)  # B, L, A
        r = -d.min(-1).values.sum(-1)
        dd = torch.cdist(self.pos, self.pos) + torch.eye(self.N, device=self.dev) * 10
        r = r - 0.5 * (dd < 0.15).float().sum((1, 2)) / 2
        self.t += 1
        return self._obs(), r, self.t >= self.T


class SpeakerListener(Base):
    """Asymmetric (by environment): speaker sees goal id only and emits a token; listener sees landmarks + token and moves."""

    def __init__(self, B, device, L=3, V=5):
        super().__init__(B, device)
        self.L, self.V, self.n_agents = L, V, 2
        self.obs_dims = [L, 4 + 2 * L + V]
        self.act_dims = [V, 5]

    def reset(self):
        B, L, dev = self.B, self.L, self.dev
        self.pos = torch.rand(B, 2, device=dev) * 2 - 1
        self.vel = torch.zeros(B, 2, device=dev)
        self.lm = torch.rand(B, L, 2, device=dev) * 2 - 1
        self.goal = torch.randint(0, L, (B,), device=dev)
        self.msg = torch.zeros(B, self.V, device=dev)
        self.t = 0
        return self._obs()

    def _obs(self):
        g = torch.nn.functional.one_hot(self.goal, self.L).float()
        lo = torch.cat([self.vel, self.pos, (self.lm - self.pos[:, None]).flatten(1), self.msg], -1)
        return [g, lo]

    def step(self, acts):
        s, l = acts
        self.msg = torch.nn.functional.one_hot(s, self.V).float()
        self.pos, self.vel = self._move(self.pos[:, None], self.vel[:, None], l[:, None])
        self.pos, self.vel = self.pos[:, 0], self.vel[:, 0]
        tgt = self.lm[torch.arange(self.B), self.goal]
        r = -(self.pos - tgt).norm(dim=-1)
        self.t += 1
        return self._obs(), r, self.t >= self.T


class SplitRendezvous(Base):
    """Symmetric roles, split information: 2 agents must both reach a goal; agent i sees only its own coordinate
    of the goal (x for agent 0, y for agent 1) and must communicate it. Identical architectures/obs sizes.
    Both agents move and both talk (V tokens)."""

    def __init__(self, B, device, V=8):
        super().__init__(B, device)
        self.V, self.n_agents = V, 2
        d = 1 + 4 + 2 + V
        self.obs_dims = [d, d]
        self.act_dims = [5 * V, 5 * V]

    def reset(self):
        B, dev = self.B, self.dev
        self.pos = torch.rand(B, 2, 2, device=dev) * 2 - 1
        self.vel = torch.zeros(B, 2, 2, device=dev)
        self.goal = torch.rand(B, 2, device=dev) * 2 - 1
        self.msg = torch.zeros(B, 2, self.V, device=dev)
        self.t = 0
        return self._obs()

    def _obs(self):
        out = []
        for i in range(2):
            out.append(torch.cat([self.goal[:, i:i + 1], self.vel[:, i], self.pos[:, i],
                                  self.pos[:, 1 - i] - self.pos[:, i], self.msg[:, 1 - i]], -1))
        return out

    def step(self, acts):
        a = torch.stack(acts, 1)
        mv, tok = a % 5, a // 5
        self.msg = torch.nn.functional.one_hot(tok, self.V).float()
        self.pos, self.vel = self._move(self.pos, self.vel, mv)
        r = -(self.pos - self.goal[:, None]).norm(dim=-1).sum(-1)
        self.t += 1
        return self._obs(), r, self.t >= self.T


def teacher(k, gain, device, seed=123):
    g = torch.Generator().manual_seed(seed)
    W = [torch.randn(k, 32, generator=g) * gain / k ** .5, torch.randn(32, 32, generator=g) * gain / 32 ** .5,
         torch.randn(32, 2, generator=g) / 32 ** .5]
    W = [w.to(device) for w in W]
    return lambda z: torch.tanh(torch.tanh(torch.tanh(z @ W[0]) @ W[1]) @ W[2] * 2)


class Leader(Base):
    """Symmetric agents, full shared info, capacity-hungry goal: all N agents see latent z (k-dim) and must all
    reach goal = f(z), f a fixed random deep MLP. Each also sees teammates' relative positions/velocities.
    Hypothesis: collaboration lets ONE agent compute f(z) while others just follow it (cheap)."""

    def __init__(self, B, device, N=2, k=8, gain=1.5):
        super().__init__(B, device)
        self.N = self.n_agents = N; self.k = k
        self.f = teacher(k, gain, device)
        d = k + 4 + 4 * (N - 1)
        self.obs_dims, self.act_dims = [d] * N, [5] * N

    def reset(self):
        B, N, dev = self.B, self.N, self.dev
        self.z = torch.randn(B, self.k, device=dev)
        self.goal = self.f(self.z)
        self.pos = torch.rand(B, N, 2, device=dev) * 2 - 1
        self.vel = torch.zeros(B, N, 2, device=dev)
        self.t = 0
        return self._obs()

    def _obs(self):
        out = []
        for i in range(self.N):
            rel = [torch.cat([self.pos[:, j] - self.pos[:, i], self.vel[:, j]], -1) for j in range(self.N) if j != i]
            out.append(torch.cat([self.z, self.vel[:, i], self.pos[:, i]] + rel, -1))
        return out

    def step(self, acts):
        a = torch.stack(acts, 1)
        self.pos, self.vel = self._move(self.pos, self.vel, a)
        r = -(self.pos - self.goal[:, None]).norm(dim=-1).sum(-1)
        self.t += 1
        return self._obs(), r, self.t >= self.T


class Leader3(Leader):
    def __init__(self, B, device): super().__init__(B, device, N=3)


ENVS = {"leader": Leader, "leader3": Leader3,"spread": Spread, "speaker_listener": SpeakerListener, "split_rdv": SplitRendezvous}
ENVS["leader1"] = lambda B, d: Leader(B, d, N=1)
