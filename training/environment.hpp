// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

#ifndef GFOOTBALL_TRAINING_ENVIRONMENT_HPP
#define GFOOTBALL_TRAINING_ENVIRONMENT_HPP

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>

class GameEnv;
struct ScenarioConfig;

constexpr int kTrainingPlayersPerTeam = 11;
constexpr int kTrainingStickyActionCount = 10;
constexpr int kTrainingActionCount = 32;

struct TrainingPlayer {
  std::array<float, 3> position{};
  std::array<float, 3> direction{};
  float tired_factor = 0.0f;
  int role = 0;
  bool has_card = false;
  bool is_active = false;
};

// Public match state copied into Python after each environment decision step.
struct TrainingObservation {
  std::array<float, 3> ball_position{};
  std::array<float, 3> ball_direction{};
  std::array<float, 3> ball_rotation{};
  std::array<std::array<TrainingPlayer, kTrainingPlayersPerTeam>, 2> teams{};
  std::array<int, 2> goals{};
  int game_mode = 0;
  int ball_owned_team = -1;
  int ball_owned_player = -1;
  int step = 0;
  std::array<bool, kTrainingStickyActionCount> sticky_actions{};
};

// Owns one headless left-agent-versus-built-in-AI training simulation.
class TrainingEnvironment {
 public:
  TrainingEnvironment(const std::filesystem::path& data_directory,
                      const std::filesystem::path& font_file,
                      int maximum_steps);
  ~TrainingEnvironment();

  TrainingObservation Reset(std::uint32_t seed);
  TrainingObservation Step(int controlled_player, int action);
  int maximum_steps() const { return maximum_steps_; }

 private:
  TrainingObservation Observe();
  std::unique_ptr<GameEnv> environment_;
  std::unique_ptr<ScenarioConfig> scenario_;
  int maximum_steps_;
};

#endif
