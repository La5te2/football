# Jackaroo

This module provides a headless single-agent environment and a PyTorch implementation of clipped Proximal Policy Optimization. The policy controls the engine-designated player on the left team against the built-in AI and selects one of 64 policy actions.

Policy actions `0..31` preserve all 32 atomic engine actions. Actions `32..63` add the four pass or shot inputs combined with eight directions. The Jackaroo adapter submits the first atomic action immediately and retains the second for the same player. Every engine step returns its observation. If that player remains designated, the retained action executes without another inference; if a different player becomes designated, the retained action and the new player's inferred action execute together in the shared 11-player decision. All other players are delegated to the built-in Eliza controllers. Team formation, tactics, set pieces, and the remaining player behavior retain the engine defaults.

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

The environment reward gives `+1` for scoring and `-1` for conceding. Non-scoring steps add a small reward for moving the ball toward the opponent's goal, `+0.01` when the left team gains possession, and `-0.01` when it loses possession. Holding possession has no recurring reward. Evaluation reports the final score without changing the policy.
