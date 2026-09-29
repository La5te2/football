# Jackaroo

Jackaroo is a single-agent football policy trained in three stages: built-in AI trajectory collection, behavior-cloning initialization, and PPO fine-tuning. The policy controls the engine-designated player against the built-in team and selects one of the engine's 32 atomic actions. Physical sides alternate between matches while every observation uses a canonical coordinate system in which the controlled team attacks toward positive $x$.

Every engine step returns the complete public model observation. The selected action is assigned to the designated player, and the remaining players use the engine's TeamAI and Eliza behavior. Formation, team behavior, set pieces, football rules, physics, and animation retain the engine defaults.

The policy represents both teams, the ball, and match context as 24 entity tokens. Three multi-head Transformer blocks model their spatial relationships, while a two-layer GRU combines the latest observation frames before separate Actor and Critic heads predict the action distribution and state value.

## Building

Build the native Python module with the Python environment containing PyTorch active on `PATH`:

```powershell
.\training\jackaroo\build.bat
```

On Linux:

```bash
bash training/jackaroo/build.sh
```

## Training

Start the complete behavior-cloning and PPO process:

```powershell
python -m training.jackaroo.train
```

On Linux, start the formal background run from the repository root:

```bash
bash training/jackaroo/run.sh
```

The script selects the system C++ runtime, validates the native environment, collects 128 built-in AI matches for two behavior-cloning epochs, collects at least 8192 transitions and one completed next-goal episode from every environment per PPO update, reuses them for eight PPO epochs, uses up to 16 parallel environments, writes the process ID to `runs/jackaroo.pid`, and writes console output to `runs/jackaroo.stdout.log`. Additional arguments override its formal defaults.

The default run collects 128 built-in AI matches, performs two behavior-cloning epochs, then starts PPO with eight concurrent native environments. The default checkpoint is `runs/jackaroo.pt` and the compact per-update log is `runs/jackaroo.pt.jsonl`.

```powershell
python -m training.jackaroo.train --imitation-games 128 --imitation-epochs 2 --updates 1000 --steps-per-update 2048 --ppo-epochs 4 --ppo-batch-size 256 --environments 8 --maximum-steps 3000 --learning-rate 0.0003 --evaluation-interval 50 --evaluation-games 2 --device auto --seed 1 --history-length 4 --checkpoint runs\jackaroo.pt
```

Use `--resume runs\jackaroo.pt` with a new `--updates` value to continue PPO from a checkpoint. Resume restores policy and optimizer state and skips behavior cloning.

Each training episode starts at kickoff and ends when either team scores the next goal or the underlying match reaches full time. Scoring, reaching full time without another goal, and conceding produce $+1$, $0$, and $-1$ respectively; every non-terminal transition has zero reward. Goals divide one continuous match into several training episodes while preserving its score, remaining time, random state, and team state. PPO uses $gamma=1$ and $lambda=1$ so a completed episode's WDL result reaches every action in that episode without temporal decay.

## Evaluation

Evaluate a checkpoint through complete deterministic matches:

```powershell
python -m training.jackaroo.evaluate runs\jackaroo.pt --games 10 --device auto --seed 1
```

Evaluation reports next-goal wins, draws, losses, mean episode length, complete-match results, and mean goal difference. Training also performs the same greedy evaluation every 50 PPO updates by default.

## Exporting

Export the final policy as a TorchScript module without optimizer or training diagnostics:

```powershell
python -m training.jackaroo.export runs\jackaroo.pt models\jackaroo\jackaroo.pt
```

The exported `.pt` contains the policy computation graph and parameters, accepts the same encoded observation history used during training, and can be loaded by a native LibTorch model plugin.
