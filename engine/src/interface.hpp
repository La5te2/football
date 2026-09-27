// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

#ifndef GFOOTBALL_INTERFACE_HPP
#define GFOOTBALL_INTERFACE_HPP

#include <cstddef>
#include <cstdint>

// Stable C ABI shared by the engine and external model DLLs. Keep this header
// independent of engine internals so model plugins can include it directly.
constexpr std::int32_t kGFootballPlayersPerTeam = 11;
constexpr std::int32_t kGFootballStickyActionCount = 10;

struct GFootballModelPlayer {
  float position[3];
  float velocity[3];
  float facing[3];
  float formation_position[2];
  float dynamic_formation_position[2];
  float tired_factor;
  std::int32_t role;
  std::int32_t dynamic_role;
  std::int32_t function_type;
  std::int32_t action_frame;
  std::int32_t touch_frame;
  std::int32_t possession_duration_ms;
  std::int32_t time_to_ball_ms;
  std::uint8_t has_card;
  std::uint8_t is_active;
  std::uint8_t touch_pending;
};

// The fourteen positioning factors already consumed by TeamAIController.
// Values are normalized to [0, 1].
struct GFootballModelTactics {
  float offense_depth_factor;
  float defense_depth_factor;
  float offense_width_factor;
  float defense_width_factor;
  float offense_own_half_factor;
  float defense_own_half_factor;
  float offense_midfield_focus;
  float defense_midfield_focus;
  float offense_midfield_focus_strength;
  float defense_midfield_focus_strength;
  float offense_side_focus_strength;
  float defense_side_focus_strength;
  float offense_micro_focus_strength;
  float defense_micro_focus_strength;
};

struct GFootballModelTeamState {
  GFootballModelTactics tactics;
  float possession_amount;
  float fading_possession_amount;
  float offside_trap_x;
  std::int32_t designated_possession_player;
  std::int32_t time_to_ball_ms;
};

// Complete input passed to a model once per 100 ms decision step. Team 0 is
// always the physical left team and team 1 the physical right team.
struct GFootballModelObservation {
  float ball_position[3];
  float ball_velocity[3];
  float ball_rotation[3];
  GFootballModelPlayer teams[2][kGFootballPlayersPerTeam];
  GFootballModelTeamState team_state[2];
  std::int32_t goals[2];
  std::int32_t game_mode;
  std::int32_t set_piece_team;
  std::int32_t set_piece_taker;
  std::int32_t ball_owned_team;
  std::int32_t ball_owned_player;
  std::int32_t last_touch_team;
  std::int32_t last_touch_player;
  std::int32_t match_time_ms;
  std::int32_t step;
  std::uint8_t is_in_play;
  // Sticky action state indexed by player. Rows delegated to Eliza are zero.
  std::uint8_t sticky_actions[kGFootballPlayersPerTeam]
                             [kGFootballStickyActionCount];
};

// The array index is the player index. game_delegate delegates that player
// to ElizaController; every other value is an atomic engine action.
struct GFootballModelDecision {
  std::int32_t actions[kGFootballPlayersPerTeam];
};

// Every model DLL exports these four functions. The engine owns observations
// and decisions; the DLL owns the opaque handle returned by create.
using GFootballModelHandle = void*;
// Creates one independent model instance for one team.
using GFootballModelCreate = GFootballModelHandle (*)(
    const char* model_directory, std::int32_t side, std::int32_t game_duration,
    char* error, std::size_t error_capacity);
using GFootballModelDestroy = void (*)(GFootballModelHandle);
// Clears recurrent state and action history before a new match.
using GFootballModelReset = void (*)(GFootballModelHandle);
// Produces one action or delegation marker for every player.
using GFootballModelDecide = std::int32_t (*)(
    GFootballModelHandle, const GFootballModelObservation*,
    GFootballModelDecision*, char* error, std::size_t error_capacity);

#endif
