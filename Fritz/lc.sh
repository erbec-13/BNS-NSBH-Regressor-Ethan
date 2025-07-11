#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --mem-bind=verbose,local
#SBATCH --cpus-per-task=16
#SBATCH --partition=ghx4
#SBATCH --time=00:20:00
#SBATCH --job-name=pytorch
#SBATCH --account=<account>
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest

export GCN_CLIENT_ID="3ipd76gggl9edibth02eb2feag"
export GCN_CLIENT_SECRET="mhl631h2hobaj0uc4d5dbr08qvp7l4fhnudjeh5mq9ja8m3vujb"
export SKYPORTAL_TOKEN="859d2347-eaed-4606-aa46-d811493d8196"

python3 main.py
