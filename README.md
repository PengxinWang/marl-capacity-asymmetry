# Capacity Asymmetry in Multi-Agent Teams: Pilot

**Question.** Does multi-agent collaboration create *agent-specific capacity asymmetry* that can be exploited for inference
efficiency? In other words, does every agent in a team really need an equally large model?

**Short answer from this pilot.** Partly yes. A team of one large agent plus small partners keeps most of the
performance of an all-large team at about half the inference cost. Small agents learn to *follow* the large one instead of
solving the hard part themselves. However, this asymmetry appears mainly when capacity is constrained **during training**;
teams trained symmetrically develop only weak, seed-dependent asymmetry on their own.

---

## Setup

### MARL benchmark: `leader` environment ([envs.py](envs.py))
- N agents with **identical** observations, actions and architecture; there are no hand-designed roles.
- Every agent sees a latent vector `z` (8-d) and must move to a goal `f(z)`, where `f` is a fixed random deep MLP. Computing `f(z)` is
  capacity-hungry: in a supervised fit, error falls monotonically with network width.
- Shared team reward: −Σ (distance of each agent to the goal), 25 steps per episode.
- Training: MAPPO ([mappo.py](mappo.py)) with a **separate actor per agent** and a centralized critic. The critic is used only in training
  and is not counted as inference cost.
- **Capacity = actor hidden width `h`** (2 hidden layers of `h` units). Each agent's width can be set independently.
- 3 seeds per configuration, 3000 iterations, batch of 512 parallel envs.

### LLM sanity check ([llm_seq.py](llm_seq.py))
- MATH-500, two Qwen2.5-Instruct agents with **identical prompts**. Agent A writes first; agent B sees A's message and gives the team answer.
- The only asymmetry is speaking order. We swap each agent's size independently (0.5B / 1.5B / 3B / 7B).
- No training, greedy decoding, n = 500 (standard error ≈ ±2 points).

---

## Main results

### 1. Large + small vs. large + large (MARL, 2 agents)
Team return: higher is better. *Retained* = share of the gap between the all-small (4/4) and all-large (64/64) teams that is recovered.

| Team (actor widths) | Total actor params | Relative inference cost | Team return (mean ± sd) | Retained |
|---|---|---|---|---|
| **Large + Large (64/64)** | 11,146 | **100%** | **−15.95 ± 0.06** | **100%** |
| Large + Medium (64/16) | 6,202 | 56% | −16.61 ± 0.22 | 93% |
| Large + Small (64/8) | 5,826 | 52% | −16.72 ± 0.11 | 92% |
| Large + Tiny (64/4) | 5,686 | 51% | −17.15 ± 0.08 | 87% |
| Medium + Medium (16/16) | 1,258 | 11% | −20.96 ± 0.36 | 47% |
| Small + Small (8/8) | 506 | 5% | −22.78 ± 0.43 | 27% |
| Tiny + Tiny (4/4) | 226 | 2% | −25.34 ± 0.13 | 0% |

### 2. Three agents
| Team | Total actor params | Relative inference cost | Team return | Retained |
|---|---|---|---|---|
| **Large ×3 (64/64/64)** | 17,487 | **100%** | **−24.76 ± 0.09** | **100%** |
| Large + Tiny + Tiny (64/4/4) | 6,087 | 35% | −27.09 ± 0.17 | 82% |
| Tiny ×3 (4/4/4) | 387 | 2% | −37.85 ± 1.40 | 0% |

### 3. Why it works: the small agent becomes a follower
Mean final distance to the goal (2 agents, lower is better):

| Agent | Distance to goal |
|---|---|
| Width-64 agent trained alone | 0.17 |
| Width-4 agent trained alone | 0.39 |
| Width-4 agent next to a width-64 teammate | **0.19** |
| … same agent, with the wide teammate frozen | 0.92 |
| Width-64 agent, with the narrow teammate frozen | 0.21 |

The small agent offloads computing `f(z)` to its teammate and just follows it. The division of labour comes from
interaction, not from an assigned role.

### 4. LLM: position alone creates asymmetric size sensitivity (MATH-500 team accuracy)
| A (writes first) \ B (final answer) | 0.5B | 1.5B | 3B | 7B |
|---|---|---|---|---|
| **0.5B** | 25.8 | 27.4 | 45.0 | 49.6 |
| **1.5B** | 45.4 | 45.6 | 55.6 | 57.6 |
| **3B** | 54.4 | 55.0 | 64.0 | 66.2 |
| **7B** | 64.4 | 64.0 | 73.0 | **73.8** |

| Team (A / B) | Total params | Relative cost | Accuracy | Δ vs 7B/7B |
|---|---|---|---|---|
| **7B / 7B** | 15.2B | **100%** | **73.8** | — |
| 7B / 3B | 10.7B | 70% | 73.0 | −0.8 |
| 3B / 7B | 10.7B | 70% | 66.2 | −7.6 |
| 7B / 0.5B | 8.1B | 53% | 64.4 | −9.4 |
| 0.5B / 7B | 8.1B | 53% | 49.6 | −24.2 |

Shrinking the first writer hurts about 2.5× more than shrinking the second, at the same cost.

### 5. Caveat: weak spontaneous asymmetry under symmetric training
When a symmetric 64/64 team is trained and we then mask 50% of one agent's hidden units, the normalized performance drops differ
by only about 1.2–3× between agents, and *which* agent is more sensitive changes with the seed. Every agent keeps part of the
`f(z)` computation, so no agent can be shrunk for free after symmetric training ([probe.py](probe.py)).

**Takeaway:** the efficiency opportunity is best pursued by *allocating or learning per-agent capacity during training*, not by
compressing a symmetric team afterwards.

### Dead ends
Classic MPE-style tasks (`spread`, `speaker_listener`, `split_rdv`) were too easy: width 4 already saturates them, so they cannot
show a capacity effect.

---

## Next steps
1. **Equal-budget comparison:** asymmetric 64/4 vs. a symmetric team with the same total parameters (≈45/45). This is the decisive efficiency test and has not been run yet.
2. **Learned capacity:** start every agent at full width with a group-sparsity penalty and see whether training discovers "one large + followers" on its own.
3. **LLM with RL:** port the follower dynamic to a multi-turn LLM environment and train it with RL.

---

## Reproduce
```bash
pip install torch            # MARL part; vllm is also needed for the LLM part
bash sweep2.sh               # leader / leader1 / leader3 sweep (GPU, ~1.5 h at 14 runs in parallel)
python summarize.py          # results table
python diag.py "runs/leader_*_s0.pt"        # follower diagnostic
python probe.py "runs/leader*_64-64*.pt"    # post-hoc masking probe
bash llm_grid.sh             # LLM 4x4 grid (uses Qwen2.5-*-Instruct from the HF cache)
```
The math grader in `llm_seq.py` is loaded from a local verl checkout; edit the path at the top of the file.

| File | Purpose |
|---|---|
| `envs.py` | Vectorized torch environments (`leader`, `leader3`, `leader1` = solo, plus the dead-end envs) |
| `mappo.py` | MAPPO with per-agent actor widths |
| `probe.py` | Post-hoc per-agent masking sensitivity |
| `diag.py` | Per-agent distance / follower diagnostic |
| `teach.py` | Calibration of the goal function's difficulty |
| `llm_seq.py`, `llm_grid.sh` | LLM size-swap grid |
| `results/` | Raw LLM grid results (`llm_grid.jsonl`) and masking probe output |
