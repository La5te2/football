# Jackaroo

This module provides a headless single-agent environment and a PyTorch implementation of clipped Proximal Policy Optimization. The policy controls the engine-designated player against the built-in AI and selects one of the engine's 32 atomic actions. Consecutive matches alternate the controlled physical side, while observations use a canonical own-team-attacks-right coordinate system.

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

The default checkpoint is `checkpoints/jackaroo.pt`. Rollout length, update count, learning rate, match length, seed, observation-history length, potential bootstrap games, potential refresh interval, and checkpoint path can be configured from the command line:

```powershell
python -m training.jackaroo.train --updates 1000 --steps-per-update 2048 --maximum-steps 3000 --learning-rate 0.0003 --device auto --seed 1 --history-length 4 --potential-bootstrap-games 4 --potential-ensemble-size 3 --potential-refresh-interval 1 --checkpoint checkpoints\jackaroo.pt
```

Use `--resume checkpoints\jackaroo.pt` together with a new `--updates` value to continue from a saved policy, learned potential, and both optimizer states. The checkpoint records the observation shape and training configuration so incompatible resumes fail before simulation starts. Complete trajectory data used by the learned potential is stored in `checkpoints/jackaroo.pt.episodes.pt`, and bounded per-update diagnostics are appended to `checkpoints/jackaroo.pt.jsonl`.

The documented ablation groups use the same environment and policy capacity: `--training-group terminal` uses only the terminal result, `--training-group learned` trains the result potential without ODA labels, and `--training-group oda` enables the complete ODA-conditioned method.

## Evaluation

Evaluate a checkpoint against the same built-in opponent:

```powershell
python -m training.jackaroo.evaluate checkpoints\jackaroo.pt --games 10 --device auto --seed 1
```

The task reward is `+1`, `0`, or `-1` for a terminal win, draw, or loss. Bootstrap matches delegate all eleven players to the built-in controllers and initialize independent ODA-conditioned potential members; their averaged ODA parameters initialize the policy relation encoder. Complete matches enter disjoint train, validation, and test partitions, while training batches balance result, match phase, and score state. Member disagreement reduces shaping confidence, and validation-gated interpolation supplies one frozen potential snapshot to each PPO interval. Rollouts collect at least `--steps-per-update` decisions and finish the active match before updating. Fixed causal observation windows provide action-phase, pass-flight, and transition context. Potential training options are available through the `--potential-*` arguments. Reward components and checkpoint diagnostics include task reward, potential reward, telescoping error, action distribution, shots, and possession transitions. Evaluation reports the final score without changing the policy.

## Exporting

Export the final policy as a TorchScript module without optimizer, learned-potential, trajectory, or diagnostic state:

```powershell
python -m training.jackaroo.export checkpoints\jackaroo.pt models\jackaroo\jackaroo.pt
```

The exported `.pt` contains the policy computation graph and parameters and can be loaded with `torch::jit::load()` from a native LibTorch plugin. The plugin implements Jackaroo's observation encoding, causal history, selected-player action submission, and model interface entry points. The plugin DLL and `.pt` remain separate deployable files; embedding the module into the DLL is an optional packaging step.
