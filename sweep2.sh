#!/bin/bash
# Leader env: symmetric agents, capacity-hungry shared goal. leader1 = solo capacity curve.
P=/localwork/pxwang24/miniforge3/envs/verl-agent/bin/python
export CUDA_VISIBLE_DEVICES=0
J=()
for s in 0 1 2; do
  for w in 64,64 64,4 4,64 4,4 64,8 8,8 16,16 64,16; do J+=("leader $w $s"); done
  for w in 4 8 16 64; do J+=("leader1 $w $s"); done
  for w in 64,64,64 64,4,4 4,4,4; do J+=("leader3 $w $s"); done
done
run(){ set -- $1; n=$1_$(echo $2|tr , -)_s$3; [ -f runs/$n.pt ] || $P mappo.py --env $1 --widths $2 --seed $3 --iters 3000 --out runs/$n.pt > logs/$n.log 2>&1; }
export -f run; export P
printf '%s\n' "${J[@]}" | xargs -P 14 -I{} bash -c 'run "{}"'
echo SWEEP2_DONE
