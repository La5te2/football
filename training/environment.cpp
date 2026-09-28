// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

// Headless simulation lifecycle and state extraction for Python training.

#include "environment.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "game_env.hpp"

namespace {

void SetEnvironment(const char* name, const std::string& value) {
#ifdef _WIN32
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

void AddFormation(std::vector<FormationEntry>& team, bool kickoff_team) {
  struct Entry {
    float x;
    float y;
    e_PlayerRole role;
  };
  static constexpr Entry formation[] = {
      {-1.0f, 0.0f, e_PlayerRole_GK},
      {0.0f, 0.02f, e_PlayerRole_RM},
      {0.0f, -0.02f, e_PlayerRole_CF},
      {-0.422f, -0.19576f, e_PlayerRole_LB},
      {-0.5f, -0.06356f, e_PlayerRole_CB},
      {-0.5f, 0.063559f, e_PlayerRole_CB},
      {-0.422f, 0.19576f, e_PlayerRole_RB},
      {-0.184212f, -0.10568f, e_PlayerRole_CM},
      {-0.267574f, 0.0f, e_PlayerRole_CM},
      {-0.184212f, 0.10568f, e_PlayerRole_CM},
      {-0.01f, -0.2161f, e_PlayerRole_LM},
  };
  for (int index = 0; index < kGFootballPlayersPerTeam; ++index) {
    float x = formation[index].x;
    float y = formation[index].y;
    if (!kickoff_team && index == 1) {
      x = -0.05f;
      y = 0.0f;
    } else if (!kickoff_team && index == 2) {
      x = -0.01f;
      y = 0.216102f;
    }
    team.emplace_back(x, y, formation[index].role, false, true);
  }
}

void CopyPosition(const Position& source, float destination[3]) {
  for (int axis = 0; axis < 3; ++axis) {
    destination[axis] = source.env_coord(axis);
  }
}

}  // namespace

TrainingEnvironment::TrainingEnvironment(
    const std::filesystem::path& data_directory,
    const std::filesystem::path& font_file, int maximum_steps)
    : maximum_steps_(maximum_steps) {
  if (maximum_steps <= 0) {
    throw std::invalid_argument("maximum_steps must be positive");
  }
  if (!std::filesystem::is_directory(data_directory)) {
    throw std::invalid_argument("engine data directory does not exist");
  }
  if (!std::filesystem::is_regular_file(font_file)) {
    throw std::invalid_argument("engine font file does not exist");
  }

  SetEnvironment("GFOOTBALL_DATA_DIR", data_directory.string());
  SetEnvironment("GFOOTBALL_FONT", font_file.string());

  environment_ = std::make_unique<GameEnv>();
  environment_->game_config.physics_steps_per_frame = 10;
  environment_->start_game();
  environment_->state = game_running;

  auto scenario = ScenarioConfig::make();
  scenario->left_agents = kGFootballPlayersPerTeam;
  scenario->right_agents = 0;
  scenario->real_time = false;
  AddFormation(scenario->left_team, true);
  AddFormation(scenario->right_team, false);
  scenario_ = std::make_unique<ScenarioConfig>(*scenario);
}

TrainingEnvironment::~TrainingEnvironment() = default;

// Restarts the complete match and advances through the noninteractive startup.
TrainingObservation TrainingEnvironment::Reset(std::uint32_t seed) {
  scenario_->game_engine_random_seed = seed;
  environment_->reset(*scenario_, false);
  while (true) {
    SharedInfo state;
    {
      ContextHolder context(environment_.get());
      state = environment_->get_info();
    }
    if (state.is_in_play && state.step >= 0) break;
    {
      ContextHolder context(environment_.get());
      environment_->step();
    }
  }
  return Observe();
}

// Applies one action to the engine-designated left player and advances one
// 100 ms decision step.
TrainingObservation TrainingEnvironment::Step(int action) {
  if (action < game_idle || action >= game_delegate) {
    throw std::out_of_range("action must be in [0, 31]");
  }
  {
    ContextHolder context(environment_.get());
    const SharedInfo state = environment_->get_info();
    const int controlled_player = state.teams[0].designated_possession_player;
    if (state.is_in_play && controlled_player >= 0 &&
        controlled_player < kGFootballPlayersPerTeam) {
      GFootballModelDecision decision{};
      for (int player = 0; player < kGFootballPlayersPerTeam; ++player) {
        decision.actions[player] = game_delegate;
      }
      decision.actions[controlled_player] = action;
      environment_->apply_model_decision(true, decision);
    }
    environment_->step();
  }
  return Observe();
}

// Copies the public model-interface observation without adding training-only
// state. This keeps the policy input identical to an external model's input.
TrainingObservation TrainingEnvironment::Observe() {
  ContextHolder context(environment_.get());
  const SharedInfo state = environment_->get_info();
  TrainingObservation observation{};
  CopyPosition(state.ball_position, observation.ball_position);
  CopyPosition(state.ball_velocity, observation.ball_velocity);
  CopyPosition(state.ball_rotation, observation.ball_rotation);
  const std::vector<PlayerInfo>* teams[] = {&state.left_team,
                                            &state.right_team};
  for (int side = 0; side < 2; ++side) {
    if (teams[side]->size() != kGFootballPlayersPerTeam) {
      throw std::runtime_error("engine did not return eleven players per team");
    }
    for (int player = 0; player < kGFootballPlayersPerTeam; ++player) {
      const PlayerInfo& source = teams[side]->at(player);
      GFootballModelPlayer& target = observation.teams[side][player];
      CopyPosition(source.player_position, target.position);
      CopyPosition(source.player_velocity, target.velocity);
      CopyPosition(source.player_facing, target.facing);
      target.formation_position[0] = source.formation_position.env_coord(0);
      target.formation_position[1] = source.formation_position.env_coord(1);
      target.dynamic_formation_position[0] =
          source.dynamic_formation_position.env_coord(0);
      target.dynamic_formation_position[1] =
          source.dynamic_formation_position.env_coord(1);
      target.tired_factor = source.tired_factor;
      target.role = source.role;
      target.dynamic_role = source.dynamic_role;
      target.function_type = source.function_type;
      target.action_frame = source.action_frame;
      target.touch_frame = source.touch_frame;
      target.possession_duration_ms = source.possession_duration_ms;
      target.time_to_ball_ms = source.time_to_ball_ms;
      target.has_card = source.has_card;
      target.is_active = source.is_active;
      target.touch_pending = source.touch_pending;
    }

    const TeamInfo& source_team = state.teams[side];
    GFootballModelTeamState& target_team = observation.team_state[side];
    target_team.possession_amount = source_team.possession_amount;
    target_team.fading_possession_amount =
        source_team.fading_possession_amount;
    target_team.offside_trap_x = source_team.offside_trap_x;
    target_team.designated_possession_player =
        source_team.designated_possession_player;
    target_team.time_to_ball_ms = source_team.time_to_ball_ms;
  }
  observation.goals[0] = state.left_goals;
  observation.goals[1] = state.right_goals;
  observation.game_mode = state.game_mode;
  observation.set_piece_team = state.set_piece_team;
  observation.set_piece_taker = state.set_piece_taker;
  observation.ball_owned_team = state.ball_owned_team;
  observation.ball_owned_player = state.ball_owned_player;
  observation.last_touch_team = state.last_touch_team;
  observation.last_touch_player = state.last_touch_player;
  observation.match_time_ms = state.match_time_ms;
  observation.step = state.step;
  observation.is_in_play = state.is_in_play;
  static constexpr int sticky_actions[] = {
      game_left, game_top_left, game_top, game_top_right, game_right,
      game_bottom_right, game_bottom, game_bottom_left, game_sprint,
      game_dribble};
  for (int player = 0; player < kGFootballPlayersPerTeam; ++player) {
    for (int index = 0; index < kGFootballStickyActionCount; ++index) {
      observation.sticky_actions[player][index] =
          environment_->sticky_action_state(sticky_actions[index], true,
                                             player);
    }
  }
  return observation;
}
