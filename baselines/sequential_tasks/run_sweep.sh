#!/bin/bash

# Run this script as run_sweep.sh NUMBER_OF_AGENTS SWEEP_ID
# It will run $1 agents on the same sweep.

export WANDB_AGENT_DISABLE_FLAPPING=true
export WANDB_AGENT_MAX_INITIAL_FAILURES=1000

for i in $(seq $1); do
  wandb agent $2 & # &> /dev/null &
done

handler() {
  pkill -KILL -P $$
}

trap handler SIGINT
# If a SIGINT is received (CTRL+C), send a SIGKILL to every agent. This is NOT a graceful termination and runs will crash.
# A cleaner option would be to send a SIGINT to simulate a CTRL+C received directly by wandb agent, however this does
# not work. As a result, the only way to prematurely terminate a sweep without crashing runs, is to control it from the
# W&B web dashboard.

wait