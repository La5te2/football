// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

#ifndef GFOOTBALL_TRAINING_ENVIRONMENT_HPP
#define GFOOTBALL_TRAINING_ENVIRONMENT_HPP

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>

#include "interface.hpp"

class GameEnv;
class ReplayWriter;
struct ScenarioConfig;

using TrainingObservation = GFootballModelObservation;
using TrainingObservationPair = std::array<TrainingObservation, 2>;
using TrainingDecision =
    std::array<std::int32_t, kGFootballPlayersPerTeam>;
using TrainingDecisionPair = std::array<TrainingDecision, 2>;

// Owns one headless simulation used for single-agent evaluation or self-play.
class TrainingEnvironment {
 public:
  TrainingEnvironment(const std::filesystem::path& data_directory,
                      const std::filesystem::path& font_file,
                      int maximum_steps);
  ~TrainingEnvironment();

  TrainingObservation Reset(std::uint32_t seed, bool left_team);
  TrainingObservationPair ResetSelfPlay(std::uint32_t seed);
  TrainingObservation Step(const TrainingDecision& actions);
  TrainingObservationPair StepSelfPlay(const TrainingDecisionPair& actions);
  void StartRecording(const std::filesystem::path& path);
  void FinishRecording();
  int maximum_steps() const { return maximum_steps_; }

 private:
  void ResetMatch(std::uint32_t seed, int left_agents, int right_agents);
  GFootballModelDecision ApplyDecision(bool left_team,
                                       const TrainingDecision& actions);
  TrainingObservation Observe(bool left_team);
  bool left_team_ = true;
  std::unique_ptr<GameEnv> environment_;
  std::unique_ptr<ScenarioConfig> scenario_;
  std::unique_ptr<ReplayWriter> replay_writer_;
  bool recording_game_active_ = false;
  int maximum_steps_;
};

#endif
