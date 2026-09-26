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
  float direction[3];
  float tired_factor;
  std::int32_t role;
  std::uint8_t has_card;
  std::uint8_t is_active;
};

// Complete input passed to a model once per 100 ms decision step. Team 0 is
// always the physical left team and team 1 the physical right team.
struct GFootballModelObservation {
  float ball_position[3];
  float ball_direction[3];
  float ball_rotation[3];
  GFootballModelPlayer teams[2][kGFootballPlayersPerTeam];
  std::int32_t goals[2];
  std::int32_t game_mode;
  std::int32_t ball_owned_team;
  std::int32_t ball_owned_player;
  std::int32_t step;
  std::uint8_t sticky_actions[kGFootballStickyActionCount];
};

// Output from one model decision: the direct recipient of this action and one
// atomic engine action from gfootball_actions.h. Player choice is not encoded
// as an action; the model names the recipient independently on every decision.
struct GFootballModelDecision {
  std::int32_t controlled_player;
  std::int32_t action;
};

// Every model DLL exports these four functions. The engine owns observations
// and decisions; the DLL owns the opaque handle returned by create.
using GFootballModelHandle = void*;
// Creates one independent model instance for one team controller.
using GFootballModelCreate = GFootballModelHandle (*)(
    const char* model_directory, std::int32_t side, std::int32_t game_duration,
    char* error, std::size_t error_capacity);
using GFootballModelDestroy = void (*)(GFootballModelHandle);
// Clears recurrent state and action history before a new match.
using GFootballModelReset = void (*)(GFootballModelHandle);
// Produces one player selection and atomic action; returns nonzero on success.
using GFootballModelDecide = std::int32_t (*)(
    GFootballModelHandle, const GFootballModelObservation*,
    GFootballModelDecision*, char* error, std::size_t error_capacity);

#endif
