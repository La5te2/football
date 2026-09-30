// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

// Headless simulation lifecycle and state extraction for Python training.

#include "environment.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "game_env.hpp"
#include "replay.hpp"

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

int OppositeDirection(int action) {
  static constexpr int directions[] = {
      game_right, game_bottom_right, game_bottom, game_bottom_left,
      game_left, game_top_left, game_top, game_top_right};
  return action >= game_left && action <= game_bottom_left
      ? directions[action - game_left]
      : action;
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
TrainingObservation TrainingEnvironment::Reset(std::uint32_t seed,
                                               bool left_team) {
  left_team_ = left_team;
  ResetMatch(seed, left_team ? kGFootballPlayersPerTeam : 0,
             left_team ? 0 : kGFootballPlayersPerTeam);
  return Observe(left_team_);
}

TrainingObservationPair TrainingEnvironment::ResetSelfPlay(std::uint32_t seed) {
  ResetMatch(seed, kGFootballPlayersPerTeam, kGFootballPlayersPerTeam);
  return {Observe(true), Observe(false)};
}

void TrainingEnvironment::ResetMatch(std::uint32_t seed, int left_agents,
                                     int right_agents) {
  if (recording_game_active_) {
    throw std::runtime_error("cannot reset before the recorded match ends");
  }
  scenario_->left_agents = left_agents;
  scenario_->right_agents = right_agents;
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
  if (replay_writer_) {
    replay_writer_->BeginGame(seed, environment_->get_state(""));
    recording_game_active_ = true;
  }
}

GFootballModelDecision TrainingEnvironment::ApplyDecision(
    bool left_team, const TrainingDecision& actions) {
  const SharedInfo state = environment_->get_info();
  const auto& physical_team = left_team ? state.left_team : state.right_team;
  GFootballModelDecision decision{};
  for (int index = 0; index < kGFootballPlayersPerTeam; ++index) {
    const int action = actions[index];
    if (action < game_idle || action > game_delegate) {
      throw std::out_of_range("decision action must be in [0, 32]");
    }
    if (action != game_delegate && !physical_team.at(index).is_active) {
      throw std::invalid_argument("decision targets an inactive player");
    }
    decision.actions[index] = left_team ? action : OppositeDirection(action);
  }
  environment_->apply_model_decision(left_team, decision);
  return decision;
}

// Applies one complete model-interface decision and advances one 100 ms step.
TrainingObservation TrainingEnvironment::Step(const TrainingDecision& actions) {
  {
    ContextHolder context(environment_.get());
    const SharedInfo before = environment_->get_info();
    if (before.is_in_play) {
      const GFootballModelDecision decision =
          ApplyDecision(left_team_, actions);
      if (replay_writer_) {
        ReplayStep step;
        step.step = before.step;
        for (auto& team : step.decisions) {
          std::fill(std::begin(team.actions), std::end(team.actions),
                    game_delegate);
        }
        step.decisions[left_team_ ? 0 : 1] = decision;
        replay_writer_->Record(step);
      }
    }
    environment_->step();
  }
  TrainingObservation observation = Observe(left_team_);
  if (recording_game_active_ && observation.step >= maximum_steps_) {
    ContextHolder context(environment_.get());
    const SharedInfo final_state = environment_->get_info();
    replay_writer_->FinishGame(
        final_state.step,
        {final_state.left_goals, final_state.right_goals});
    recording_game_active_ = false;
  }
  return observation;
}

TrainingObservationPair TrainingEnvironment::StepSelfPlay(
    const TrainingDecisionPair& actions) {
  {
    ContextHolder context(environment_.get());
    if (environment_->get_info().is_in_play) {
      ApplyDecision(true, actions[0]);
      ApplyDecision(false, actions[1]);
    }
    environment_->step();
  }
  return {Observe(true), Observe(false)};
}

void TrainingEnvironment::StartRecording(const std::filesystem::path& path) {
  if (replay_writer_) {
    throw std::runtime_error("recording is already active");
  }
  replay_writer_ = std::make_unique<ReplayWriter>(path);
}

void TrainingEnvironment::FinishRecording() {
  if (!replay_writer_) {
    throw std::runtime_error("recording has not started");
  }
  if (recording_game_active_) {
    throw std::runtime_error("cannot finish recording before the match ends");
  }
  replay_writer_->Finish();
}

// Copies the public model-interface observation without adding training-only
// state. This keeps the policy input identical to an external model's input.
TrainingObservation TrainingEnvironment::Observe(bool left_team) {
  ContextHolder context(environment_.get());
  const SharedInfo state = environment_->get_info();
  TrainingObservation observation{};
  const int own_side = left_team ? 0 : 1;
  const int opponent_side = 1 - own_side;
  const float rotation = left_team ? 1.0f : -1.0f;
  auto copy_canonical = [rotation](const Position& source,
                                   float destination[3]) {
    destination[0] = rotation * source.env_coord(0);
    destination[1] = rotation * source.env_coord(1);
    destination[2] = source.env_coord(2);
  };
  copy_canonical(state.ball_position, observation.ball_position);
  copy_canonical(state.ball_velocity, observation.ball_velocity);
  copy_canonical(state.ball_rotation, observation.ball_rotation);
  const std::vector<PlayerInfo>* teams[] = {&state.left_team,
                                            &state.right_team};
  const int canonical_side[] = {own_side, opponent_side};
  for (int side = 0; side < 2; ++side) {
    const int physical_side = canonical_side[side];
    if (teams[physical_side]->size() != kGFootballPlayersPerTeam) {
      throw std::runtime_error("engine did not return eleven players per team");
    }
    for (int player = 0; player < kGFootballPlayersPerTeam; ++player) {
      const PlayerInfo& source = teams[physical_side]->at(player);
      GFootballModelPlayer& target = observation.teams[side][player];
      copy_canonical(source.player_position, target.position);
      copy_canonical(source.player_velocity, target.velocity);
      copy_canonical(source.player_facing, target.facing);
      target.formation_position[0] =
          rotation * source.formation_position.env_coord(0);
      target.formation_position[1] =
          rotation * source.formation_position.env_coord(1);
      target.dynamic_formation_position[0] =
          rotation * source.dynamic_formation_position.env_coord(0);
      target.dynamic_formation_position[1] =
          rotation * source.dynamic_formation_position.env_coord(1);
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

    const TeamInfo& source_team = state.teams[physical_side];
    GFootballModelTeamState& target_team = observation.team_state[side];
    target_team.possession_amount = source_team.possession_amount;
    target_team.fading_possession_amount =
        source_team.fading_possession_amount;
    target_team.offside_trap_x = rotation * source_team.offside_trap_x;
    target_team.designated_possession_player =
        source_team.designated_possession_player;
    target_team.time_to_ball_ms = source_team.time_to_ball_ms;
  }
  observation.goals[0] = left_team ? state.left_goals : state.right_goals;
  observation.goals[1] = left_team ? state.right_goals : state.left_goals;
  observation.game_mode = state.game_mode;
  observation.set_piece_team = state.set_piece_team < 0
      ? -1
      : (state.set_piece_team == own_side ? 0 : 1);
  observation.set_piece_taker = state.set_piece_taker;
  observation.ball_owned_team = state.ball_owned_team < 0
      ? -1
      : (state.ball_owned_team == own_side ? 0 : 1);
  observation.ball_owned_player = state.ball_owned_player;
  observation.last_touch_team = state.last_touch_team < 0
      ? -1
      : (state.last_touch_team == own_side ? 0 : 1);
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
      int physical_action = sticky_actions[index];
      if (!left_team) physical_action = OppositeDirection(physical_action);
      observation.sticky_actions[player][index] =
          environment_->sticky_action_state(physical_action, left_team,
                                             player);
    }
  }
  return observation;
}
