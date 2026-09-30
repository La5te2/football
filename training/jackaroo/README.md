# Jackaroo

Jackaroo is a single-agent recurrent PPO policy for the next-goal objective. It controls the engine-designated player with one of the engine's 32 atomic actions while TeamAI and Eliza control the remaining players. Observations use canonical coordinates, so Jackaroo always attacks toward positive $x$ regardless of its physical side.

The policy starts from random parameters and learns from scored segments produced by direct interaction with the engine. A segment begins at kickoff and ends at the next goal. Scoring and conceding produce $+1$ and $-1$ respectively; a scoreless final segment is retained in evaluation statistics and excluded from optimization.

The network combines a 24-token entity Transformer, persistent per-entity and global recurrent state, a DSS-derived spatial-control field, a structured representation of the 32 actions, an action-conditioned transition model, and one next-goal EPV function. The same EPV is the PPO Critic and evaluates predicted action outcomes; there is no separate Q network.

## Building

Build the native Python environment with the Python installation containing PyTorch active on `PATH`:

```powershell
.\training\jackaroo\build.bat
```

On Linux:

```bash
bash training/jackaroo/build.sh
```

## Training

Start a local training run:

```powershell
python -m training.jackaroo.train
```

Start the formal Linux background run from the repository root:

```bash
bash training/jackaroo/run.sh
```

The formal script uses up to 16 concurrent native environments, collects at least 8192 transitions from complete scored segments for each update, trains on contiguous sequences of 32 transitions, and performs four PPO epochs. It writes the process ID to `runs/jackaroo.pid`, the checkpoint to `runs/jackaroo.pt`, and console output to `runs/jackaroo.stdout.log`. Extra command-line arguments override the script defaults.

```powershell
python -m training.jackaroo.train --updates 1000 --steps-per-update 2048 --sequence-length 32 --ppo-epochs 4 --ppo-batch-size 256 --environments 8 --maximum-steps 3000 --learning-rate 0.0001 --entropy-coefficient 0.001 --transition-coefficient 0.1 --action-value-coefficient 0.1 --control-coefficient 0.05 --evaluation-interval 50 --evaluation-games 2 --device auto --seed 1 --checkpoint runs\jackaroo.pt
```

One run reuses `--seed` for every training match and alternates physical sides. Independent fixed seeds are used for evaluation. Use `--resume runs\jackaroo.pt` with a new `--updates` value to continue from a compatible checkpoint.

## Evaluation

Evaluate a checkpoint through complete matches:

```powershell
python -m training.jackaroo.evaluate runs\jackaroo.pt --games 10 --device auto --seed 1
```

Evaluation reports next-goal wins, draws, losses, mean segment length, complete-match results, and mean goal difference.

## Exporting

Export the recurrent policy as a TorchScript module:

```powershell
python -m training.jackaroo.export runs\jackaroo.pt models\jackaroo\jackaroo.pt
```

The exported module accepts one encoded public observation plus persistent entity and global states, then returns action logits, EPV, and the updated states for native inference.
