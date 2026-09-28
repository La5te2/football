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
struct ScenarioConfig;

using TrainingObservation = GFootballModelObservation;

// Owns one headless left-model-versus-built-in-AI training simulation.
class TrainingEnvironment {
 public:
  TrainingEnvironment(const std::filesystem::path& data_directory,
                      const std::filesystem::path& font_file,
                      int maximum_steps);
  ~TrainingEnvironment();

  TrainingObservation Reset(std::uint32_t seed);
  TrainingObservation Step(int action);
  int maximum_steps() const { return maximum_steps_; }

 private:
  TrainingObservation Observe();
  std::unique_ptr<GameEnv> environment_;
  std::unique_ptr<ScenarioConfig> scenario_;
  int maximum_steps_;
};

#endif
