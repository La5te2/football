// Copyright 2019 Google LLC & Bastiaan Konings
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// written by bastiaan konings schuiling 2008 - 2015
// this work is public domain. the code is undocumented, scruffy, untested, and should generally not be used for anything important.
// i do not offer support, so don't ask. to be used for inspiration :)

#ifndef _HPP_FOOTBALL_ONTHEPITCH_EXTERNAL_EXECUTOR
#define _HPP_FOOTBALL_ONTHEPITCH_EXTERNAL_EXECUTOR

#include "playercontroller.hpp"

#include "../../../ai/external_input.hpp"

class Player;

class ExternalExecutor : public PlayerController {

  public:
    ExternalExecutor(Match *match = nullptr, ExternalInput *input = nullptr);
    virtual ~ExternalExecutor();

    virtual void SetPlayer(PlayerBase *player);
    bool Disabled() const {
      return input->Disabled();
    }

    virtual void RequestCommand(PlayerCommandQueue &commandQueue);
    virtual void Process();
    virtual Vector3 GetDirection();
    virtual float GetFloatVelocity();

    void PreProcess(Match *match, ExternalInput *input) {
      this->match = match;
      this->input = input;
   }

    void ProcessState(EnvState* state) { DO_VALIDATION;
      ProcessPlayerController(state);
      input->ProcessState(state);
      state->process(actionMode);
      state->process(actionButton);
      state->process(actionBufferTime_ms);
      state->process(gauge_ms);
      state->process(previousDirection);
      state->process(steadyDirection);
      state->process(lastSteadyDirectionSnapshotTime_ms);
    }
    virtual int GetReactionTime_ms();

    ExternalInput *GetInput() { return input; }

    int GetActionMode() { DO_VALIDATION; return actionMode; }

    virtual void Reset();

  protected:

    void _GetExternalInput(Vector3 &rawInputDirection, float &rawInputVelocityFloat);

    ExternalInput *input;

    // set when a contextual button (example: pass/defend button) is pressed
    // once this is set and the button stays pressed, it stays the same
    // 0: undefined, 1: off-the-ball button active, 2: on-the-ball button active/action queued
    int actionMode = 0;

    e_ButtonFunction actionButton;
    int actionBufferTime_ms = 0;
    int gauge_ms = 0;

    // Preserve the original analog-style direction smoothing used by GRF.
    Vector3 previousDirection;
    Vector3 steadyDirection;
    int lastSteadyDirectionSnapshotTime_ms = 0;
    float mirror = 1.0;
};

#endif
