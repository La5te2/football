// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

// Serialization for deterministic native match recordings.

#include "replay.hpp"

#include <array>
#include <stdexcept>
#include <type_traits>

#include "gfootball_actions.h"

namespace {

constexpr std::array<char, 8> kGameMagic = {
    'M', 'A', 'T', 'C', 'H', 'G', 'F', 'R'};
constexpr std::uint64_t kMaximumStateSize = 256ull * 1024ull * 1024ull;
constexpr std::int32_t kMaximumStepCount = 10000000;
constexpr std::int32_t kMaximumGameCount = 1000000;

template <typename T>
void WriteValue(std::ostream& stream, const T& value) {
  static_assert(std::is_arithmetic_v<T>);
  stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!stream) throw std::runtime_error("Unable to write replay file");
}

template <typename T>
T ReadValue(std::istream& stream) {
  static_assert(std::is_arithmetic_v<T>);
  T value{};
  stream.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!stream) throw std::runtime_error("Replay file is truncated");
  return value;
}

bool IsValidDecision(const GFootballModelDecision& decision) {
  for (std::int32_t action : decision.actions) {
    if (action < game_idle || action > game_delegate) return false;
  }
  return true;
}

void WriteStep(std::ostream& stream, const ReplayStep& step) {
  WriteValue(stream, step.step);
  for (const auto& decision : step.decisions) {
    for (std::int32_t action : decision.actions) WriteValue(stream, action);
  }
}

ReplayStep ReadStep(std::istream& stream) {
  ReplayStep step;
  step.step = ReadValue<std::int32_t>(stream);
  for (auto& decision : step.decisions) {
    for (std::int32_t& action : decision.actions) {
      action = ReadValue<std::int32_t>(stream);
    }
    if (!IsValidDecision(decision)) {
      throw std::runtime_error("Replay contains an invalid player action");
    }
  }
  return step;
}

}  // namespace

ReplayWriter::ReplayWriter(const std::filesystem::path& path)
    : stream_(path, std::ios::binary | std::ios::trunc) {
  if (!stream_) {
    throw std::runtime_error("Unable to create replay file: " + path.string());
  }
}

ReplayWriter::~ReplayWriter() = default;

void ReplayWriter::BeginGame(std::uint32_t seed,
                             const std::string& initial_state) {
  if (finished_) throw std::runtime_error("Replay recording is already finished");
  if (game_active_) throw std::runtime_error("A replay game is already active");
  if (game_count_ >= kMaximumGameCount) {
    throw std::runtime_error("Replay contains too many games");
  }
  if (initial_state.empty() || initial_state.size() > kMaximumStateSize) {
    throw std::runtime_error("Invalid replay game state");
  }
  stream_.write(kGameMagic.data(), kGameMagic.size());
  if (!stream_) throw std::runtime_error("Unable to write replay game header");
  WriteValue(stream_, seed);
  current_step_count_offset_ = stream_.tellp();
  WriteValue(stream_, std::int32_t{0});
  current_final_step_offset_ = stream_.tellp();
  WriteValue(stream_, std::int32_t{0});
  current_goals_offset_ = stream_.tellp();
  WriteValue(stream_, std::int32_t{0});
  WriteValue(stream_, std::int32_t{0});
  WriteValue(stream_, static_cast<std::uint64_t>(initial_state.size()));
  stream_.write(initial_state.data(),
                static_cast<std::streamsize>(initial_state.size()));
  if (!stream_) throw std::runtime_error("Unable to write replay game state");
  current_step_count_ = 0;
  last_recorded_step_ = -1;
  game_active_ = true;
}

void ReplayWriter::Record(const ReplayStep& step) {
  if (finished_) throw std::runtime_error("Replay recording is already finished");
  if (!game_active_) throw std::runtime_error("No replay game is active");
  if ((current_step_count_ == 0 && step.step != 0) ||
      (current_step_count_ > 0 && step.step != last_recorded_step_ + 1) ||
      current_step_count_ >= kMaximumStepCount) {
    throw std::runtime_error("Replay decision steps are not contiguous");
  }
  for (const auto& decision : step.decisions) {
    if (!IsValidDecision(decision)) {
      throw std::runtime_error("Replay contains an invalid player action");
    }
  }
  WriteStep(stream_, step);
  ++current_step_count_;
  last_recorded_step_ = step.step;
}

void ReplayWriter::FinishGame(
    std::int32_t final_step, const std::array<std::int32_t, 2>& goals) {
  if (!game_active_) throw std::runtime_error("No replay game is active");
  if (current_step_count_ <= 0 || final_step != last_recorded_step_ + 1 ||
      final_step > kMaximumStepCount || goals[0] < 0 || goals[1] < 0) {
    throw std::runtime_error("Invalid replay game result");
  }
  const std::streampos end = stream_.tellp();
  stream_.seekp(current_step_count_offset_);
  WriteValue(stream_, current_step_count_);
  stream_.seekp(current_final_step_offset_);
  WriteValue(stream_, final_step);
  stream_.seekp(current_goals_offset_);
  WriteValue(stream_, goals[0]);
  WriteValue(stream_, goals[1]);
  stream_.seekp(end);
  game_active_ = false;
  ++game_count_;
}

void ReplayWriter::Finish() {
  if (finished_) return;
  if (game_active_) throw std::runtime_error("Cannot finish an active replay game");
  if (game_count_ == 0) throw std::runtime_error("Replay contains no games");
  stream_.flush();
  if (!stream_) throw std::runtime_error("Unable to finalize replay file");
  finished_ = true;
}

ReplayReader::ReplayReader(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Unable to open replay file: " + path.string());
  }
  while (stream.peek() != std::char_traits<char>::eof()) {
    if (games_.size() >= static_cast<std::size_t>(kMaximumGameCount)) {
      throw std::runtime_error("Replay contains too many games");
    }
    std::array<char, kGameMagic.size()> game_magic{};
    stream.read(game_magic.data(), game_magic.size());
    if (!stream || game_magic != kGameMagic) {
      throw std::runtime_error("Invalid replay game boundary");
    }

    ReplayGame game;
    game.seed = ReadValue<std::uint32_t>(stream);
    const std::int32_t step_count = ReadValue<std::int32_t>(stream);
    game.final_step = ReadValue<std::int32_t>(stream);
    game.goals[0] = ReadValue<std::int32_t>(stream);
    game.goals[1] = ReadValue<std::int32_t>(stream);
    const std::uint64_t state_size = ReadValue<std::uint64_t>(stream);
    if (step_count <= 0 || step_count > kMaximumStepCount ||
        game.final_step <= 0 || game.final_step > kMaximumStepCount ||
        game.goals[0] < 0 || game.goals[1] < 0 || state_size == 0 ||
        state_size > kMaximumStateSize) {
      throw std::runtime_error("Invalid replay game metadata");
    }

    game.initial_state.resize(static_cast<std::size_t>(state_size));
    stream.read(game.initial_state.data(),
                static_cast<std::streamsize>(state_size));
    if (!stream) throw std::runtime_error("Replay game state is truncated");

    game.steps.reserve(static_cast<std::size_t>(step_count));
    for (std::int32_t step_index = 0; step_index < step_count; ++step_index) {
      ReplayStep step = ReadStep(stream);
      if ((step_index == 0 && step.step != 0) ||
          (step_index > 0 && step.step != game.steps.back().step + 1)) {
        throw std::runtime_error("Replay decision steps are not contiguous");
      }
      game.steps.push_back(std::move(step));
    }
    if (game.final_step != game.steps.back().step + 1) {
      throw std::runtime_error("Replay game result does not match its steps");
    }
    games_.push_back(std::move(game));
  }

  if (games_.empty()) throw std::runtime_error("Replay contains no games");
}
