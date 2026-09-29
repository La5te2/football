# Jackaroo

Jackaroo is a single-agent football policy trained in three stages: built-in AI trajectory collection, behavior-cloning initialization, and PPO fine-tuning. The policy controls the engine-designated player against the built-in team and selects one of the engine's 32 atomic actions. Matches alternate the controlled physical side while every observation uses a canonical coordinate system in which the controlled team attacks toward positive $x$.

Every engine step returns the complete public model observation. The selected action is assigned to the designated player, and the remaining players use the engine's TeamAI and Eliza behavior. Formation, team behavior, set pieces, football rules, physics, and animation retain the engine defaults.

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

The default run collects eight built-in AI matches, performs three behavior-cloning epochs, then starts PPO with eight concurrent native environments. The default checkpoint is `checkpoints/jackaroo.pt` and the compact per-update log is `checkpoints/jackaroo.pt.jsonl`.

```powershell
python -m training.jackaroo.train --imitation-games 8 --imitation-epochs 3 --updates 1000 --steps-per-update 2048 --environments 8 --maximum-steps 3000 --learning-rate 0.0003 --potential-scale 0.20 --evaluation-interval 50 --evaluation-games 2 --device auto --seed 1 --history-length 4 --checkpoint checkpoints\jackaroo.pt
```

Use `--resume checkpoints\jackaroo.pt` with a new `--updates` value to continue PPO from a checkpoint. Resume restores policy and optimizer state and skips behavior cloning.

The task reward contains goal-difference increments and the terminal match result. A fixed bounded football potential supplies policy-invariant stepwise shaping with $gamma=1$ and terminal potential zero. The potential rewards territory, controlled possession, and goal threat as progress signals; its episode sum depends only on the initial state. Training diagnostics report task reward, shaping reward, telescoping error, PPO losses, entropy, and action distribution.

## Evaluation

Evaluate a checkpoint through complete deterministic matches:

```powershell
python -m training.jackaroo.evaluate checkpoints\jackaroo.pt --games 10 --device auto --seed 1
```

Evaluation reports wins, draws, losses, and mean goal difference. Training also performs the same greedy evaluation every 50 PPO updates by default. These match outcomes determine policy quality independently of shaping reward.

## Exporting

Export the final policy as a TorchScript module without optimizer or training diagnostics:

```powershell
python -m training.jackaroo.export checkpoints\jackaroo.pt models\jackaroo\jackaroo.pt
```

The exported `.pt` contains the policy computation graph and parameters. A native LibTorch plugin supplies observation encoding, causal history, selected-player action submission, and model interface entry points.
