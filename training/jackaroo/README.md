# Jackaroo

This module provides a headless single-agent environment and a PyTorch implementation of clipped Proximal Policy Optimization. The policy controls the engine-designated player on the left team against the built-in AI and selects one of the engine's 32 atomic actions.

Every engine step returns its observation. The selected action is assigned to the engine-designated player and every other player is delegated to the built-in Eliza controllers. Team formation, tactics, set pieces, and the remaining player behavior retain the engine defaults.

## Building

Build the native Python module with the Python interpreter containing PyTorch active on `PATH`:

```powershell
.\training\jackaroo\build.bat
```

On Linux, run:

```bash
./training/jackaroo/build.sh
```

## Training

Start training with the default 3000-step match length:

```powershell
python -m training.jackaroo.train
```

The default checkpoint is `checkpoints/jackaroo.pt`. Rollout length, update count, learning rate, match length, seed, and checkpoint path can be configured from the command line:

```powershell
python -m training.jackaroo.train --updates 1000 --steps-per-update 2048 --maximum-steps 3000 --learning-rate 0.0003 --device auto --seed 1 --checkpoint checkpoints\jackaroo.pt
```

## Evaluation

Evaluate a checkpoint against the same built-in opponent:

```powershell
python -m training.jackaroo.evaluate checkpoints\jackaroo.pt --games 10 --device auto --seed 1
```

The environment reward gives `+1` for scoring and `-1` for conceding. Dense shaping uses the discounted change in a team-state potential composed of possession, time-to-ball advantage, available passing options, local numerical superiority, team structure, and goal threat. Small event rewards recognize a completed pass initiated by the policy and a quick counterpress recovery. Failed passes and turnovers are penalized according to their danger near the team's own goal, while an unreceived pass receives a smaller penalty. Individual components are available in `info["reward_components"]` for diagnosis. Evaluation reports the final score without changing the policy.
