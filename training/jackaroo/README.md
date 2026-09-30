# Jackaroo

Jackaroo is a single-agent recurrent PPO policy for the next-goal objective. It controls the engine-designated player with one of the engine's 32 atomic actions while TeamAI and Eliza control the remaining players. Observations use canonical coordinates, so Jackaroo always attacks toward positive $x$ regardless of its physical side.

The policy starts from random parameters and learns through shared-policy self-play. A segment begins at kickoff and ends at the next goal. The scoring side receives $+1$, the conceding side receives $-1$, and both trajectories enter PPO as separate state-dependent action sequences. A scoreless final segment remains part of its complete match and is excluded from optimization.

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

The formal script runs flights of 16 concurrent matches. Each update collects 256 complete self-play matches, lets the shared Jackaroo policy decide for both teams at every engine step, retains every scored segment, trains on contiguous sequences of 32 transitions for four PPO epochs, and then plays two complete validation matches against built-in AI with seed 42 from opposite physical sides. It writes validation replays to `runs/validation`, the process ID to `runs/jackaroo.pid`, the checkpoint to `runs/jackaroo.pt`, and console output to `runs/jackaroo.stdout.log`. Extra command-line arguments override the script defaults.

```powershell
python -m training.jackaroo.train --updates 1000 --games-per-update 256 --sequence-length 32 --ppo-epochs 4 --ppo-batch-size 256 --flight 16 --maximum-steps 3000 --learning-rate 0.0001 --entropy-coefficient 0.001 --transition-coefficient 0.1 --action-value-coefficient 0.1 --control-coefficient 0.05 --validation-seed 42 --validation-directory runs\validation --device auto --seed 1 --checkpoint runs\jackaroo.pt
```

`--seed` initializes a deterministic sequence of unique 32-bit match seeds, so a run is reproducible while every self-play match receives its own seed. Validation always uses `--validation-seed` twice, once for each physical side. Use `--resume runs\jackaroo.pt` with a new `--updates` value to continue from a compatible checkpoint.

## Evaluation

Evaluate a checkpoint through complete matches:

```powershell
python -m training.jackaroo.evaluate runs\jackaroo.pt --games 10 --device auto --seed 1
```

Evaluation reports complete-match wins, draws, losses, scores, and mean goal difference. Add `--record runs\evaluation.gfr` to save all evaluated matches in one replay file.

## Exporting

Export the recurrent policy as a TorchScript module:

```powershell
python -m training.jackaroo.export runs\jackaroo.pt models\jackaroo\jackaroo.pt
```

The exported module accepts one encoded public observation plus persistent entity and global states, then returns action logits, EPV, and the updated states for native inference.
