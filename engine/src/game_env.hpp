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

#ifndef _GAME_ENV
#define _GAME_ENV

#include <filesystem>
#include <memory>

#include "onthepitch/match.hpp"
#include "gamedefines.hpp"
#include "gfootball_actions.h"
#include "main.hpp"
#include "replay.hpp"

class ExternalInput;
class GameTask;
class Model;

typedef std::vector<std::string> StringVector;

class ContextHolder {
 public:
  ContextHolder(GameEnv* game) : game(game) {
     SetGame(game);
     GetGraphicsSystem()->SetContext();
  }
  ~ContextHolder() {
    if (GetGame() != game) {
      Log(e_FatalError, "football", "main", "game state was corrupted");
    }
    GetGraphicsSystem()->DisableContext();
  }
 private:
  const GameEnv* game;
};

// Owns one simulation instance and its attached external models.
struct GameEnv {
  GameEnv();
  ~GameEnv();
  // Start the game (in separate process).
  void start_game();

  // Get the current state of the game (observation).
  SharedInfo get_info();

  // Get the current rendered frame.
  screenshoot get_frame();

  bool window_closed() const { return quit_requested_; }

  // Executes the action inside the game.
  bool sticky_action_state(int action, bool left_team, int player);
  void action(int action, bool left_team, int player);
  bool set_controlled_player(bool left_team, int controller, int player);
  void load_model(const std::filesystem::path& path, bool left_team,
                  int controller, int game_duration);
  void start_recording(const std::filesystem::path& path,
                       bool left_external, bool right_external,
                       int match_duration);
  void begin_recording_game(std::uint32_t seed,
                            const std::string& initial_state);
  void finish_recording_game(const SharedInfo& final_state);
  void finish_recording();
  void start_replay(const ReplayGame& game);
  bool replay_complete() const;
  // Resets every loaded external model instance on both teams.
  void reset_models();
  // Selects the existing engine random stream for one match.
  void set_random_seed(std::uint32_t seed);
  void record_model_decision(
      bool left_team, const GFootballModelObservation& observation,
      const GFootballModelDecision& decision);
  void reset(ScenarioConfig& game_config, bool init_animation);
  void render(bool swap_buffer = true);
  std::string get_state(const std::string& pickle);
  std::string set_state(const std::string& state);
  void step();
  void ProcessState(EnvState* state);
  ScenarioConfig& config();

 private:
  void setConfig(ScenarioConfig& scenario_config);
  void poll_window_events();
  void wait_for_playback();
  void do_step(int count);
  void getObservations();
  bool disable_graphics_ = false;
  int last_step_rendered_frames_ = 1;
  bool quit_requested_ = false;
  bool paused_ = false;
  bool timing_reset_requested_ = false;
  int playback_rate_index_ = 2;
  std::vector<std::unique_ptr<Model>> models_;
  std::unique_ptr<ReplayWriter> replay_writer_;
  const ReplayGame* replay_game_ = nullptr;
  std::size_t replay_step_index_ = 0;
  ReplayStep recording_step_;
  bool recording_step_active_ = false;
 public:
  ScenarioConfig scenario_config;
  GameConfig game_config;
  GameContext* context = nullptr;
  GameState state = game_created;
};

#endif
