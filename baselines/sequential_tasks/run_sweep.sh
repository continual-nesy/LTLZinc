#!/bin/bash

# Run this script as run_sweep.sh NUMBER_OF_CPU_AGENTS SWEEP_ID
# It will run $1 + 1 agents on the same sweep, one of them on the original device and the remainder on CPU.

export WANDB_AGENT_DISABLE_FLAPPING=true
export WANDB_AGENT_MAX_INITIAL_FAILURES=1000

# Execute the first agent with the original device (hopefully cuda:x).
wandb agent $2 &
pids[1]=$! # Store pid of last forked process.

# And execute $1 agents, but after overriding device to CPU.
# Luckily Scallop runs on CPU anyways and MLPs are small, so we can assume a limited performance drop.
# There is however a chance of timeout for GRUs. In that case it is best to re-run timed out experiments on GPU.

export DEVICE=cpu

pids=()
for i in $(seq $1); do
  wandb agent $2 &
  pids[$(($i + 1))]=$!

done

# In case of SIGINT (CTRL-C), send a SIG??? to every subprocess.
#send_signal() {
#  for pid in ${pids[*]}; do
#    TODO: This signal is not caught by W&B! Probably because it launches a subprocess with a different pid.
#           Even if it was a sigkill, the process would still be running.
#           A different process (eg. sleep 60) would catch this signal without problems.
#    kill -s SIGINT $pid
#  done
#}
#
#trap 'send_signal' SIGINT


# Wait for every subprocess to finish.
for pid in ${pids[*]}; do
  wait $pid
done