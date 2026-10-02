# Jackaroo

Jackaroo is a single-agent recurrent PPO policy for the next-goal objective. It controls the engine-designated player with one of the engine's 32 atomic actions while TeamAI and Eliza control the remaining players. Observations use canonical coordinates, so Jackaroo always attacks toward positive $x$ regardless of its physical side.

Training begins with pretraining on a prepared TamakEri self-play dataset and then continues through shared-policy Jackaroo self-play. The preparation command runs the engine once and writes compressed HDF5 sequences. GPU pretraining subsequently reads this dataset without running TamakEri or the football engine. A segment begins at kickoff and ends at the next goal. The scoring side receives $+1$, the conceding side receives $-1$, and both trajectories enter policy and value training as separate state-dependent action sequences. A scoreless final segment supplies behavior and auxiliary transition supervision while remaining outside the next-goal return.

The network combines a 24-token entity Transformer, persistent per-entity and global recurrent state, direction-aware arrival times, continuous team control, state-dependent spatial quality, a structured representation of the 32 actions, an action-conditioned transition model, and one next-goal value function. The PPO Critic and action-value calculation share this value function.

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

Install the HDF5 reader in the active Python environment:

```powershell
python -m pip install h5py
```

Generate the reusable compressed TamakEri dataset:

```powershell
python -m training.jackaroo.prepare --games 256 --flight 16 --maximum-steps 3000 --sequence-length 32 --device auto --seed 1 --output runs\tamakeri.h5
```

The preparation stage is CPU-engine bound and can run separately from training. Increase `--games` to enlarge the dataset. Each team trajectory is stored as ordered 32-step chunks. Running the command again replaces the selected output file.

Pretrain the network on the GPU and save a standalone checkpoint. The loader shuffles complete team trajectories, advances their chunks in causal order, and carries detached entity and global GRU states across chunk boundaries for truncated backpropagation through time:

```powershell
python -m training.jackaroo.pretrain --data runs\tamakeri.h5 --checkpoint runs\jackaroo.pt --epochs 2 --batch-size 4096 --device auto
```

Continue from that checkpoint with online self-play PPO:

```powershell
python -m training.jackaroo.train --resume runs\jackaroo.pt --checkpoint runs\jackaroo.pt --device auto
```

`prepare`, `pretrain`, and `train` are independent programs. `pipeline.sh` runs them in order and skips preparation or pretraining when their output already exists.

Start the formal Linux background run from the repository root:

```bash
bash training/jackaroo/run.sh
```

The formal script generates `runs/tamakeri.h5` and the pretrained checkpoint when they are needed, then collects 256 complete Jackaroo self-play matches per PPO update. Native matches advance asynchronously and completed simulations return to the next inference batch immediately. Prepared teacher tails supply behavior and auxiliary supervision. Online PPO tails supply auxiliary supervision. Both forms remain outside the next-goal policy and value losses. Each update finishes with two complete validation matches against built-in AI using seed 42 from opposite physical sides. The script writes validation replays to `runs/validation`, the process ID to `runs/jackaroo.pid`, the checkpoint to `runs/jackaroo.pt`, and console output to `runs/jackaroo.stdout.log`.

```powershell
python -m training.jackaroo.train --resume runs\jackaroo.pt --updates 1000 --games-per-update 256 --sequence-length 32 --ppo-epochs 4 --ppo-batch-size 256 --flight 16 --maximum-steps 3000 --learning-rate 0.0001 --entropy-coefficient 0.001 --transition-coefficient 0.1 --action-value-coefficient 0.1 --control-coefficient 0.05 --space-coefficient 0.05 --validation-seed 42 --validation-directory runs\validation --device auto --seed 1 --checkpoint runs\jackaroo.pt
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

The exported module accepts one encoded public observation plus persistent entity and global states, then returns action logits, next-goal value, and the updated states for native inference.
