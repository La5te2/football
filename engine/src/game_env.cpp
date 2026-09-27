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

#undef NDEBUG

#include "game_env.hpp"

#include <algorithm>
#include <array>
#include <fenv.h>

#include <cerrno>
#include <chrono>
#include <ctime>
#include <ratio>
#include <thread>

#include <SDL2/SDL.h>

#include "ai/external_input.hpp"
#include "file.h"
#include "gametask.hpp"
#include "model.hpp"

using std::string;

namespace {

constexpr std::array<double, 4> kPlaybackRates = {0.25, 0.5, 1.0, 2.0};

}  // namespace

GameEnv::GameEnv() { DO_VALIDATION; }
GameEnv::~GameEnv() = default;

void GameEnv::do_step(int count) {
  DO_VALIDATION;
  while (count--) {
    DO_VALIDATION;
    wait_for_playback();
    if (quit_requested_) return;
    context->gameTask->ProcessPhase();
  }
}

float Position::env_coord(int index) const {
  switch (index) {
    DO_VALIDATION;
    case 0:
      return value[0] / X_FIELD_SCALE;
    case 1:
      return value[1] / Y_FIELD_SCALE;
    case 2:
      return value[2] / Z_FIELD_SCALE;
    default:
      Log(e_FatalError, "football", "main", "index out of range");
      return 0;
  }
}

std::string Position::debug() const {
  DO_VALIDATION;
  return std::to_string(value[0]) + "," + std::to_string(value[1]) + "," +
         std::to_string(value[2]);
}

void GameEnv::setConfig(ScenarioConfig& scenario_config) {
  DO_VALIDATION;
  scenario_config.ball_position.coords[0] =
      scenario_config.ball_position.coords[0] * X_FIELD_SCALE;
  scenario_config.ball_position.coords[1] =
      scenario_config.ball_position.coords[1] * Y_FIELD_SCALE;
  std::vector<SideSelection> setup = GetMenuTask()->GetSideSelection();
  GFOOTBALL_CHECK(setup.size() == 2 * MAX_PLAYERS);
  int controller = 0;
  for (int x = 0; x < scenario_config.left_agents; x++) {
    DO_VALIDATION;
    setup[controller++].side = -1;
  }
  while (controller < MAX_PLAYERS) {
    DO_VALIDATION;
    setup[controller++].side = 0;
  }
  for (int x = 0; x < scenario_config.right_agents; x++) {
    DO_VALIDATION;
    setup[controller++].side = 1;
  }
  while (controller < 2 * MAX_PLAYERS) {
    DO_VALIDATION;
    setup[controller++].side = 0;
  }
  this->scenario_config = scenario_config;
  GetMenuTask()->SetSideSelection(setup);
}

void GameEnv::start_game() {
  assert(context == nullptr);
  install_stacktrace();
  context = new GameContext();
  ContextHolder c(this);
  // feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);

  char* data_dir = getenv("GFOOTBALL_DATA_DIR");
  if (data_dir) {
    DO_VALIDATION;
    GetGameConfig().data_dir = data_dir;
  }
  Properties* config = new Properties();
  config->Set("match_duration", 0.027);
  char* font_file = getenv("GFOOTBALL_FONT");
  if (font_file) {
    DO_VALIDATION;
    config->Set("font_filename", font_file);
  }
  config->Set("game", 0);
  run_game(config, game_config.render);
  auto scenario_config = ScenarioConfig::make();
  reset(*scenario_config, false);
  DO_VALIDATION;
}

SharedInfo GameEnv::get_info() {
  SharedInfo info;
  GetGameTask()->GetMatch()->GetState(&info);
  info.step = context->step;
  return info;
}

Screenshot GameEnv::get_frame() {
  SetGame(this);
  return GetGraphicsSystem()->GetScreen();
}

void GameEnv::poll_window_events() {
  if (!game_config.render || context == nullptr ||
      SDL_WasInit(SDL_INIT_VIDEO) == 0) {
    return;
  }

  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_QUIT ||
        (event.type == SDL_WINDOWEVENT &&
         event.window.event == SDL_WINDOWEVENT_CLOSE)) {
      quit_requested_ = true;
    } else if (event.type == SDL_KEYDOWN && !event.key.repeat) {
      switch (event.key.keysym.sym) {
        case SDLK_SPACE:
          paused_ = !paused_;
          if (!paused_) timing_reset_requested_ = true;
          break;
        case SDLK_LEFTBRACKET:
          if (playback_rate_index_ > 0) {
            --playback_rate_index_;
            timing_reset_requested_ = true;
          }
          break;
        case SDLK_RIGHTBRACKET:
          if (playback_rate_index_ + 1 <
              static_cast<int>(kPlaybackRates.size())) {
            ++playback_rate_index_;
            timing_reset_requested_ = true;
          }
          break;
      }
    }
  }
}

void GameEnv::wait_for_playback() {
  if (!game_config.render) return;
  poll_window_events();
  while (paused_ && !quit_requested_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    poll_window_events();
  }
}

bool GameEnv::sticky_action_state(int action, bool left_team, int player) {
  SetGame(this);
  int input_id = player + (left_team ? 0 : 11);
  auto input = GetExternalInputs()[input_id];
  switch (Action(action)) {
    case game_left:
      return input->GetOriginalDirection() == Vector3(-1, 0, 0);
    case game_top_left:
      return input->GetOriginalDirection() == Vector3(-1, 1, 0);
    case game_top:
      return input->GetOriginalDirection() == Vector3(0, 1, 0);
    case game_top_right:
      return input->GetOriginalDirection() == Vector3(1, 1, 0);
    case game_right:
      return input->GetOriginalDirection() == Vector3(1, 0, 0);
    case game_bottom_right:
      return input->GetOriginalDirection() == Vector3(1, -1, 0);
    case game_bottom:
      return input->GetOriginalDirection() == Vector3(0, -1, 0);
    case game_bottom_left:
      return input->GetOriginalDirection() == Vector3(-1, -1, 0);
    case game_keeper_rush:
      return input->GetButton(e_ButtonFunction_KeeperRush);
    case game_pressure:
      return input->GetButton(e_ButtonFunction_Pressure);
    case game_team_pressure:
      return input->GetButton(e_ButtonFunction_TeamPressure);
    case game_sprint:
      return input->GetButton(e_ButtonFunction_Sprint);
    case game_dribble:
      return input->GetButton(e_ButtonFunction_Dribble);
    default:
      Log(e_FatalError, "football", "main", "invalid sticky action");
  }
  return false;
}

bool GameEnv::set_controlled_player(bool left_team, int controller,
                                    int player) {
  SetGame(this);
  Team* team = GetGameTask()->GetMatch()->GetTeam(left_team ? 0 : 1);
  return team->SetExternalControllerPlayer(controller, player);
}

void GameEnv::apply_model_decision(
    bool left_team, const GFootballModelDecision& decision) {
  for (int player = 0; player < kGFootballPlayersPerTeam; ++player) {
    const int player_action = decision.actions[player];
    if (player_action == game_delegate) {
      action(game_delegate, left_team, player);
      continue;
    }
    if (!set_controlled_player(left_team, player, player)) {
      throw std::runtime_error("Unable to assign model action recipient");
    }
    action(player_action, left_team, player);
  }
}

void GameEnv::load_model(const std::filesystem::path& path, bool left_team,
                         int game_duration) {
  models_.push_back(std::make_unique<Model>(path, left_team, game_duration));
}

void GameEnv::start_recording(const std::filesystem::path& path) {
  if (replay_game_) {
    throw std::runtime_error("Cannot record while replaying a match");
  }
  replay_writer_ = std::make_unique<ReplayWriter>(path);
}

void GameEnv::begin_recording_game(std::uint32_t seed,
                                   const std::string& initial_state) {
  if (replay_writer_) replay_writer_->BeginGame(seed, initial_state);
}

void GameEnv::finish_recording_game(const SharedInfo& final_state) {
  if (!replay_writer_) return;
  replay_writer_->FinishGame(
      final_state.step,
      std::array<std::int32_t, 2>{final_state.left_goals,
                                  final_state.right_goals});
}

void GameEnv::finish_recording() {
  if (replay_writer_) replay_writer_->Finish();
}

void GameEnv::start_replay(const ReplayGame& game) {
  if (!models_.empty() || replay_writer_) {
    throw std::runtime_error("Replay requires an environment without models");
  }
  replay_game_ = &game;
  replay_step_index_ = 0;
}

bool GameEnv::replay_complete() const {
  return replay_game_ && replay_step_index_ == replay_game_->steps.size();
}

void GameEnv::reset_models() {
  for (auto& model : models_) model->Reset();
}

void GameEnv::set_random_seed(std::uint32_t seed) {
  SetGame(this);
  scenario_config.game_engine_random_seed = seed;
  randomize(seed);
}

void GameEnv::record_model_decision(
    bool left_team, const GFootballModelDecision& decision) {
  if (!replay_writer_) return;
  if (!recording_step_active_) {
    throw std::runtime_error("Model decision occurred outside a replay step");
  }
  const int side = left_team ? 0 : 1;
  recording_step_.decisions[side] = decision;
}

void GameEnv::action(int action, bool left_team, int player) {
  SetGame(this);
  int input_id = player + (left_team ? 0 : 11);
  auto input = GetExternalInputs()[input_id];
  input->SetDisabled(false);
  switch (Action(action)) {
    case game_idle:
      break;
    case game_left:
      input->SetDirection(Vector3(-1, 0, 0));
      break;
    case game_top_left:
      input->SetDirection(Vector3(-1, 1, 0));
      break;
    case game_top:
      input->SetDirection(Vector3(0, 1, 0));
      break;
    case game_top_right:
      input->SetDirection(Vector3(1, 1, 0));
      break;
    case game_right:
      input->SetDirection(Vector3(1, 0, 0));
      break;
    case game_bottom_right:
      input->SetDirection(Vector3(1, -1, 0));
      break;
    case game_bottom:
      input->SetDirection(Vector3(0, -1, 0));
      break;
    case game_bottom_left:
      input->SetDirection(Vector3(-1, -1, 0));
      break;

    case game_long_pass:
      input->SetButton(e_ButtonFunction_LongPass, true);
      break;
    case game_high_pass:
      input->SetButton(e_ButtonFunction_HighPass, true);
      break;
    case game_short_pass:
      input->SetButton(e_ButtonFunction_ShortPass, true);
      break;
    case game_shot:
      input->SetButton(e_ButtonFunction_Shot, true);
      break;
    case game_keeper_rush:
      input->SetButton(e_ButtonFunction_KeeperRush, true);
      break;
    case game_sliding:
      input->SetButton(e_ButtonFunction_Sliding, true);
      break;
    case game_pressure:
      input->SetButton(e_ButtonFunction_Pressure, true);
      break;
    case game_team_pressure:
      input->SetButton(e_ButtonFunction_TeamPressure, true);
      break;
    case game_attacking_run:
      input->SetButton(e_ButtonFunction_AttackingRun, true);
      break;
    case game_sprint:
      input->SetButton(e_ButtonFunction_Sprint, true);
      break;
    case game_dribble:
      input->SetButton(e_ButtonFunction_Dribble, true);
      break;
    case game_release_direction:
      input->SetDirection(Vector3(0, 0, 0));
      break;
    case game_release_long_pass:
      input->SetButton(e_ButtonFunction_LongPass, false);
      break;
    case game_release_high_pass:
      input->SetButton(e_ButtonFunction_HighPass, false);
      break;
    case game_release_short_pass:
      input->SetButton(e_ButtonFunction_ShortPass, false);
      break;
    case game_release_shot:
      input->SetButton(e_ButtonFunction_Shot, false);
      break;
    case game_release_keeper_rush:
      input->SetButton(e_ButtonFunction_KeeperRush, false);
      break;
    case game_release_sliding:
      input->SetButton(e_ButtonFunction_Sliding, false);
      break;
    case game_release_pressure:
      input->SetButton(e_ButtonFunction_Pressure, false);
      break;
    case game_release_team_pressure:
      input->SetButton(e_ButtonFunction_TeamPressure, false);
      break;
    case game_release_attacking_run:
      input->SetButton(e_ButtonFunction_AttackingRun, false);
      break;
    case game_release_sprint:
      input->SetButton(e_ButtonFunction_Sprint, false);
      break;
    case game_release_dribble:
      input->SetButton(e_ButtonFunction_Dribble, false);
      break;
    case game_delegate:
      input->SetDisabled(true);
      break;
  }
}

std::string GameEnv::get_state(const std::string& pickle) {
  ContextHolder c(this);
  EnvState reader(this, "");
  string mutable_pickle = pickle;
  reader.process(mutable_pickle);
  ProcessState(&reader);
  return reader.GetState();
}

std::string GameEnv::set_state(const std::string& state) {
  SetGame(this);
  EnvState writer(this, state);
  string pickle;
  writer.process(pickle);
  ProcessState(&writer);
  if (!writer.eos()) {
    Log(e_FatalError, "football", "main", "corrupted state");
  }
  return pickle;
}

void GameEnv::step() {
  DO_VALIDATION;
  wait_for_playback();
  if (quit_requested_) return;
  timing_reset_requested_ = false;
  if (context->gameTask->GetMatch()->IsInPlay()) {
    const SharedInfo model_state = get_info();
    if (replay_game_) {
      if (replay_step_index_ >= replay_game_->steps.size()) {
        throw std::runtime_error("Replay ended before the match state");
      }
      const ReplayStep& replay_step =
          replay_game_->steps.at(replay_step_index_++);
      if (replay_step.step != model_state.step) {
        throw std::runtime_error("Replay step does not match the engine state");
      }
      for (int side = 0; side < 2; ++side) {
        const GFootballModelDecision& decision = replay_step.decisions[side];
        const bool left_team = side == 0;
        apply_model_decision(left_team, decision);
      }
    } else if (replay_writer_) {
      recording_step_ = ReplayStep{};
      recording_step_.step = model_state.step;
      for (auto& decision : recording_step_.decisions) {
        std::fill(std::begin(decision.actions), std::end(decision.actions),
                  game_delegate);
      }
      recording_step_active_ = true;
      try {
        for (auto& model : models_) {
          model->Decide(*this, model_state);
        }
        replay_writer_->Record(recording_step_);
        recording_step_active_ = false;
      } catch (...) {
        recording_step_active_ = false;
        throw;
      }
    } else {
      for (auto& model : models_) {
        model->Decide(*this, model_state);
      }
    }
  }
  // We do 10 environment steps per second, while game does 100 frames of
  // physics animation.
  int steps_to_do = GetGameConfig().physics_steps_per_frame;
  if (GetScenarioConfig().real_time) {
    DO_VALIDATION;
    int configured_render_frames = GetGameConfig().render_frames_per_step;
    int rendered_frames = configured_render_frames > 0
        ? std::min(configured_render_frames, steps_to_do)
        : last_step_rendered_frames_;
    auto start = std::chrono::steady_clock::now();
    for (int x = 1; x <= steps_to_do; x++) {
      DO_VALIDATION;
      do_step(1);
      if (quit_requested_) break;
      const double rate = kPlaybackRates[playback_rate_index_];
      const auto target_per_frame = std::chrono::duration<double, std::milli>(
          10.0 / rate);
      if (timing_reset_requested_) {
        start = std::chrono::steady_clock::now() -
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    target_per_frame * (x - 1));
        timing_reset_requested_ = false;
      }
      bool render_current_step =
          x * rendered_frames / steps_to_do !=
          (x - 1) * rendered_frames / steps_to_do;
      if (GetGameConfig().render && render_current_step) {
        render();
      }
      if (configured_render_frames > 0) {
        std::this_thread::sleep_until(
            start +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                target_per_frame * x));
      }
    }
    if (configured_render_frames <= 0) {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start);
      const double target_ms = 10.0 / kPlaybackRates[playback_rate_index_];
      if (elapsed.count() > target_ms * (steps_to_do + 1) &&
          last_step_rendered_frames_ > 1) {
        DO_VALIDATION;
        last_step_rendered_frames_--;
      } else if (elapsed.count() < target_ms * (steps_to_do - 1) &&
                 last_step_rendered_frames_ < steps_to_do) {
        DO_VALIDATION;
        last_step_rendered_frames_++;
      }
    }
  } else {
    do_step(steps_to_do);
    if (GetGameConfig().render && !quit_requested_) {
      render();
    }
  }
  if (quit_requested_) return;
  if (context->gameTask->GetMatch()->IsInPlay()) {
    DO_VALIDATION;
    context->step++;
    for (auto input : GetExternalInputs()) {
      DO_VALIDATION;
      input->ResetNotSticky();
    }
  }
}

void GameEnv::ProcessState(EnvState* state) {
  state->process(this->state);
  context->ProcessState(state);
  context->gameTask->GetMatch()->ProcessState(state);
}

void GameEnv::render(bool swap_buffer) {
  context->gameTask->PrepareRender();
  context->graphicsSystem.GetTask()->Render(swap_buffer);
}

void GameEnv::reset(ScenarioConfig& game_config, bool animations) {
  DO_VALIDATION;
  ContextHolder c(this);
  context->step = -1;
  quit_requested_ = false;
  paused_ = false;
  timing_reset_requested_ = false;
  playback_rate_index_ = 2;
  setConfig(game_config);
  for (auto input : GetExternalInputs()) {
    DO_VALIDATION;
    input->SetDisabled(true);
  }
  context->geometry_manager.RemoveUnused();
  context->surface_manager.RemoveUnused();
  context->texture_manager.RemoveUnused();
  context->vertices_manager.RemoveUnused();
  GetMenuTask()->GetWindowManager()->GetRoot()->SetRecursiveZPriority(0);
  DO_VALIDATION;
  GetMenuTask()->GetWindowManager()->GetPagePath()->Clear();
  bool already_loaded = GetGameTask()->StopMatch();
  GetMenuTask()->SetMatchData(new MatchData());
  if (!already_loaded) {
    // We show loading page only the first time when env. is started.
    GetMenuTask()->GetWindowManager()->GetPageFactory()->CreatePage(1, 0);
  }
  if (GetGameConfig().render) {
    context->graphicsSystem.GetTask()->Render(true);
  }
  GetGameTask()->StartMatch(animations);
  for (auto& model : models_) {
    model->Reset();
  }
}
