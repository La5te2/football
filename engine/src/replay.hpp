// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

#ifndef GFOOTBALL_REPLAY_HPP
#define GFOOTBALL_REPLAY_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "interface.hpp"

// One decision step stored in a native match recording.
struct ReplayStep {
  std::int32_t step = 0;
  std::array<GFootballModelDecision, 2> decisions{};
};

// One complete match within a replay file.
struct ReplayGame {
  std::uint32_t seed = 0;
  std::string initial_state;
  std::int32_t final_step = 0;
  std::array<std::int32_t, 2> goals{};
  std::vector<ReplayStep> steps;
};

class ReplayWriter final {
 public:
  explicit ReplayWriter(const std::filesystem::path& path);
  ~ReplayWriter();
  ReplayWriter(const ReplayWriter&) = delete;
  ReplayWriter& operator=(const ReplayWriter&) = delete;

  void BeginGame(std::uint32_t seed, const std::string& initial_state);
  void Record(const ReplayStep& step);
  void FinishGame(std::int32_t final_step,
                  const std::array<std::int32_t, 2>& goals);
  void Finish();

 private:
  std::ofstream stream_;
  std::streampos current_step_count_offset_{};
  std::streampos current_final_step_offset_{};
  std::streampos current_goals_offset_{};
  std::int32_t game_count_ = 0;
  std::int32_t current_step_count_ = 0;
  std::int32_t last_recorded_step_ = -1;
  bool game_active_ = false;
  bool finished_ = false;
};

class ReplayReader final {
 public:
  explicit ReplayReader(const std::filesystem::path& path);

  std::size_t game_count() const { return games_.size(); }
  const ReplayGame& game(std::size_t index) const { return games_.at(index); }

 private:
  std::vector<ReplayGame> games_;
};

#endif
