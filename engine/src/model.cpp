// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

// Model-agnostic engine-side loader and dispatcher for external model plugins.

#include "model.hpp"

#include <array>
#include <algorithm>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <dlfcn.h>
#endif

#include "game_env.hpp"

namespace {

// Platform-specific dynamic-library operations are kept behind these helpers so
// the Model lifecycle below is identical on Windows and Unix-like systems.
void* OpenLibrary(const std::filesystem::path& path) {
#ifdef _WIN32
  return reinterpret_cast<void*>(LoadLibraryExW(
      path.c_str(), nullptr,
      LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS));
#else
  return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void CloseLibrary(void* library) {
#ifdef _WIN32
  FreeLibrary(reinterpret_cast<HMODULE>(library));
#else
  dlclose(library);
#endif
}

void* FindSymbol(void* library, const char* name) {
#ifdef _WIN32
  return reinterpret_cast<void*>(
      GetProcAddress(reinterpret_cast<HMODULE>(library), name));
#else
  return dlsym(library, name);
#endif
}

template <typename Function>
Function RequiredSymbol(void* library, const char* name) {
  auto function = reinterpret_cast<Function>(FindSymbol(library, name));
  if (!function) {
    throw std::runtime_error(std::string("Model plugin is missing ") + name);
  }
  return function;
}

void CopyPosition(const Position& source, float destination[3]) {
  for (int axis = 0; axis < 3; ++axis) {
    destination[axis] = source.env_coord(axis);
  }
}

}  // namespace

// Loads the requested plugin, resolves its C entry points, and creates an opaque
// model instance owned by the plugin.
Model::Model(const std::filesystem::path& path, bool left_team,
             int game_duration)
    : left_team_(left_team) {
  const auto absolute_path = std::filesystem::absolute(path);
  library_ = OpenLibrary(absolute_path);
  if (!library_) {
    throw std::runtime_error("Unable to load model plugin: " +
                             absolute_path.string());
  }
  try {
    const auto create = RequiredSymbol<GFootballModelCreate>(
        library_, "gfootball_model_create");
    destroy_ = RequiredSymbol<GFootballModelDestroy>(
        library_, "gfootball_model_destroy");
    reset_ = RequiredSymbol<GFootballModelReset>(
        library_, "gfootball_model_reset");
    decide_ = RequiredSymbol<GFootballModelDecide>(
        library_, "gfootball_model_decide");
    std::array<char, 1024> error{};
    model_ = create(absolute_path.parent_path().string().c_str(),
                    left_team ? 0 : 1, game_duration, error.data(),
                    error.size());
    if (!model_) {
      throw std::runtime_error(error[0] ? error.data()
                                        : "Model plugin creation failed");
    }
  } catch (...) {
    CloseLibrary(library_);
    library_ = nullptr;
    throw;
  }
}

Model::~Model() {
  if (model_) destroy_(model_);
  if (library_) CloseLibrary(library_);
}

void Model::Reset() {
  reset_(model_);
}

// Converts observable match state, requests and validates a decision, then
// submits its recipient and action through the external-control pipeline.
void Model::Decide(GameEnv& env, const SharedInfo& state) {
  GFootballModelObservation observation{};
  CopyPosition(state.ball_position, observation.ball_position);
  CopyPosition(state.ball_velocity, observation.ball_velocity);
  CopyPosition(state.ball_rotation, observation.ball_rotation);
  const std::vector<PlayerInfo>* teams[] = {&state.left_team,
                                            &state.right_team};
  for (int side = 0; side < 2; ++side) {
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
    target_team.tactics = {
        source_team.tactics.offense_depth_factor,
        source_team.tactics.defense_depth_factor,
        source_team.tactics.offense_width_factor,
        source_team.tactics.defense_width_factor,
        source_team.tactics.offense_own_half_factor,
        source_team.tactics.defense_own_half_factor,
        source_team.tactics.offense_midfield_focus,
        source_team.tactics.defense_midfield_focus,
        source_team.tactics.offense_midfield_focus_strength,
        source_team.tactics.defense_midfield_focus_strength,
        source_team.tactics.offense_side_focus_strength,
        source_team.tactics.defense_side_focus_strength,
        source_team.tactics.offense_micro_focus_strength,
        source_team.tactics.defense_micro_focus_strength};
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
          env.sticky_action_state(sticky_actions[index], left_team_, player);
    }
  }

  GFootballModelDecision decision{};
  std::fill(std::begin(decision.actions), std::end(decision.actions),
            game_delegate);
  std::array<char, 1024> error{};
  if (!decide_(model_, &observation, &decision, error.data(), error.size())) {
    throw std::runtime_error(error[0] ? error.data()
                                      : "Model plugin inference failed");
  }
  const std::vector<PlayerInfo>& own_team =
      left_team_ ? state.left_team : state.right_team;
  for (int player = 0; player < kGFootballPlayersPerTeam; ++player) {
    const int action = decision.actions[player];
    if (action < game_idle || action > game_delegate ||
        (action != game_delegate && !own_team.at(player).is_active)) {
      throw std::runtime_error("Model plugin returned an invalid action");
    }
  }
  env.record_model_decision(left_team_, decision);
  env.apply_model_decision(left_team_, decision);
}
