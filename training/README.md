# Training

This module provides a headless single-agent environment and a PyTorch implementation of clipped Proximal Policy Optimization. The policy controls the engine-designated player on the left team against the built-in AI and selects one of the 32 atomic engine actions every 100 ms.

The training environment uses the same public observation and 11-player decision structures as external model plugins. Each decision delegates ten players to the built-in Eliza controllers and assigns the PPO action only to `designated_possession_player`. Team formation, tactics, set pieces, and the remaining player behavior retain the engine defaults.

## Building

Build the native Python module with the Python interpreter containing PyTorch active on `PATH`:

```powershell
.\training\build.bat
```

On Linux, run:

```bash
./training/build.sh
```

## Training

Start training with the default 3000-step match length:

```powershell
python -m training.train
```

The default checkpoint is `checkpoints/ppo.pt`. Rollout length, update count, learning rate, match length, seed, and checkpoint path can be configured from the command line:

```powershell
python -m training.train --updates 1000 --steps-per-update 2048 --maximum-steps 3000 --learning-rate 0.0003 --device auto --seed 1 --checkpoint checkpoints\ppo.pt
```

## Evaluation

Evaluate a checkpoint against the same built-in opponent:

```powershell
python -m training.evaluate checkpoints\ppo.pt --games 10 --device auto --seed 1
```

The environment reward is the left-team goal difference increment plus small ball-progress and possession terms. Evaluation reports the final score without changing the policy.
