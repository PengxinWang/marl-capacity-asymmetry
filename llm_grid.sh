#!/bin/bash
P=/localwork/pxwang24/miniforge3/envs/verl-agent/bin/python
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
for m in 0.5B 1.5B 3B 7B; do [ -f llm_out/A_$m.jsonl ] || $P llm_seq.py stage1 $m; done
for m in 0.5B 1.5B 3B 7B; do $P llm_seq.py stage2 $m 0.5B,1.5B,3B,7B; done
echo LLM_DONE
