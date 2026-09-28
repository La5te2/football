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

#ifndef _HPP_EXTERNAL_CONTROLLER
#define _HPP_EXTERNAL_CONTROLLER

#include "../defines.hpp"

#include "../scene/scene3d/scene3d.hpp"

#include "player/controller/external_executor.hpp"
#include "external_input.hpp"

using namespace blunted;

class Team;

// One member of a team's external-controller set. The team creates N
// controllers, where N is ScenarioConfig::left_agents or right_agents, and
// together they control N selected players at any instant. N is 1 in the
// single-agent configuration. Each controller owns an ExternalInput action
// stream and an ExternalExecutor, and records which player that executor
// currently drives. Changing selection rebinds the controller to another
// player. Controllers remain assigned to their team when field ends switch.
class ExternalController {

  public:
    ExternalController(Team *team, ExternalInput *input);
    ExternalController() {}
    ExternalController(const ExternalController&) = delete;
    void operator=(const ExternalController&) = delete;
    ~ExternalController();

    Player *GetSelectedPlayer() const {
      return selectedPlayer;
    }
    void SetSelectedPlayer(Player* player);
    ExternalInput *GetInput() { DO_VALIDATION; return input; }
    ExternalExecutor* GetExternalExecutor() { DO_VALIDATION; return &executor; }
    void ProcessState(EnvState *state);

  protected:
    Team *team = nullptr;
    ExternalInput *input = nullptr;
    ExternalExecutor executor;
    Player *selectedPlayer = nullptr;
};

#endif
