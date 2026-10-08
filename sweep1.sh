#!/bin/bash
# Exp1: full-width teams (for masking probes). Exp2: one agent narrowed at train time (true capacity requirement).
P=/localwork/pxwang24/miniforge3/envs/verl-agent/bin/python
export CUDA_VISIBLE_DEVICES=0
jobs_list=()
for s in 0 1 2; do
  jobs_list+=("spread 64,64,64 $s" "speaker_listener 64,64 $s" "split_rdv 64,64 $s")
  for w in 4 8; do
    jobs_list+=("spread $w,64,64 $s" "spread 64,$w,64 $s" "spread 64,64,$w $s")
    jobs_list+=("speaker_listener $w,64 $s" "speaker_listener 64,$w $s")
    jobs_list+=("split_rdv $w,64 $s" "split_rdv 64,$w $s")
  done
  jobs_list+=("spread 8,8,8 $s" "speaker_listener 8,8 $s" "split_rdv 8,8 $s")
done
run(){ set -- $1; n=$1_$(echo $2|tr , -)_s$3; [ -f runs/$n.pt ] || $P mappo.py --env $1 --widths $2 --seed $3 --iters 2000 --out runs/$n.pt > logs/$n.log 2>&1; }
export -f run; export P
printf '%s\n' "${jobs_list[@]}" | xargs -P 12 -I{} bash -c 'run "{}"'
echo SWEEP1_DONE
