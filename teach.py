"""Calibrate the capacity-hungry goal function used by the Leader env: supervised fit vs student width."""
import torch, torch.nn as nn
from envs import teacher

torch.manual_seed(0)
for gain in [1.5, 3.0]:
    f = teacher(8, gain, "cpu")
    z = torch.randn(20000, 8); y = f(z); print("gain", gain, "y std", y.std(0))
    for h in [2, 4, 8, 16, 64]:
        m = nn.Sequential(nn.Linear(8, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 2))
        o = torch.optim.Adam(m.parameters(), 3e-3)
        for i in range(3000):
            idx = torch.randint(0, 20000, (512,)); l = ((m(z[idx]) - y[idx]) ** 2).mean()
            o.zero_grad(); l.backward(); o.step()
        zt = torch.randn(5000, 8); print(h, (m(zt) - f(zt)).norm(dim=-1).mean().item())
