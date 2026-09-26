// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

// Serialization for native gfootball match recordings.

#include "replay.hpp"

#include <array>
#include <stdexcept>
#include <type_traits>

#include "gfootball_actions.h"

namespace {

constexpr std::array<char, 8> kReplayMagic = {
    'G', 'F', 'R', 'M', 'A', 'T', 'C', 'H'};
constexpr std::array<char, 8> kGameMagic = {
    'M', 'A', 'T', 'C', 'H', 'G', 'F', 'R'};
constexpr std::uint32_t kReplayFormatVersion = 1;
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

void WritePlayer(std::ostream& stream, const GFootballModelPlayer& player) {
  for (float value : player.position) WriteValue(stream, value);
  for (float value : player.direction) WriteValue(stream, value);
  WriteValue(stream, player.tired_factor);
  WriteValue(stream, player.role);
  WriteValue(stream, player.has_card);
  WriteValue(stream, player.is_active);
}

GFootballModelPlayer ReadPlayer(std::istream& stream) {
  GFootballModelPlayer player{};
  for (float& value : player.position) value = ReadValue<float>(stream);
  for (float& value : player.direction) value = ReadValue<float>(stream);
  player.tired_factor = ReadValue<float>(stream);
  player.role = ReadValue<std::int32_t>(stream);
  player.has_card = ReadValue<std::uint8_t>(stream);
  player.is_active = ReadValue<std::uint8_t>(stream);
  return player;
}

void WriteObservation(std::ostream& stream,
                      const GFootballModelObservation& observation) {
  for (float value : observation.ball_position) WriteValue(stream, value);
  for (float value : observation.ball_direction) WriteValue(stream, value);
  for (float value : observation.ball_rotation) WriteValue(stream, value);
  for (const auto& team : observation.teams) {
    for (const auto& player : team) WritePlayer(stream, player);
  }
  for (std::int32_t goal : observation.goals) WriteValue(stream, goal);
  WriteValue(stream, observation.game_mode);
  WriteValue(stream, observation.ball_owned_team);
  WriteValue(stream, observation.ball_owned_player);
  WriteValue(stream, observation.step);
  for (std::uint8_t action : observation.sticky_actions) {
    WriteValue(stream, action);
  }
}

GFootballModelObservation ReadObservation(std::istream& stream) {
  GFootballModelObservation observation{};
  for (float& value : observation.ball_position) value = ReadValue<float>(stream);
  for (float& value : observation.ball_direction) value = ReadValue<float>(stream);
  for (float& value : observation.ball_rotation) value = ReadValue<float>(stream);
  for (auto& team : observation.teams) {
    for (auto& player : team) player = ReadPlayer(stream);
  }
  for (std::int32_t& goal : observation.goals) {
    goal = ReadValue<std::int32_t>(stream);
  }
  observation.game_mode = ReadValue<std::int32_t>(stream);
  observation.ball_owned_team = ReadValue<std::int32_t>(stream);
  observation.ball_owned_player = ReadValue<std::int32_t>(stream);
  observation.step = ReadValue<std::int32_t>(stream);
  for (std::uint8_t& action : observation.sticky_actions) {
    action = ReadValue<std::uint8_t>(stream);
  }
  return observation;
}

void WriteStep(std::ostream& stream, const ReplayStep& step) {
  WriteValue(stream, step.step);
  for (int side = 0; side < 2; ++side) {
    WriteValue(stream, step.has_observation[side]);
    if (step.has_observation[side]) {
      WriteObservation(stream, step.observations[side]);
    }
  }
  for (int side = 0; side < 2; ++side) {
    WriteValue(stream, step.has_decision[side]);
    if (step.has_decision[side]) {
      WriteValue(stream, step.decisions[side].controlled_player);
      WriteValue(stream, step.decisions[side].action);
    }
  }
}

ReplayStep ReadStep(std::istream& stream) {
  ReplayStep step;
  step.step = ReadValue<std::int32_t>(stream);
  for (int side = 0; side < 2; ++side) {
    step.has_observation[side] = ReadValue<std::uint8_t>(stream);
    if (step.has_observation[side] > 1) {
      throw std::runtime_error("Invalid replay observation flag");
    }
    if (step.has_observation[side]) {
      step.observations[side] = ReadObservation(stream);
    }
  }
  for (int side = 0; side < 2; ++side) {
    step.has_decision[side] = ReadValue<std::uint8_t>(stream);
    if (step.has_decision[side] > 1) {
      throw std::runtime_error("Invalid replay decision flag");
    }
    if (step.has_decision[side]) {
      step.decisions[side].controlled_player =
          ReadValue<std::int32_t>(stream);
      step.decisions[side].action = ReadValue<std::int32_t>(stream);
    }
  }
  return step;
}

}  // namespace

// Writes a replay header and the full state from which playback begins.
ReplayWriter::ReplayWriter(
    const std::filesystem::path& path, const std::string& initial_state,
    const std::array<std::uint8_t, 2>& external_teams,
    std::int32_t match_duration)
    : stream_(path, std::ios::binary | std::ios::trunc),
      external_teams_(external_teams),
      match_duration_(match_duration) {
  if (!stream_) {
    throw std::runtime_error("Unable to create replay file: " + path.string());
  }
  if (initial_state.empty() || match_duration <= 0 ||
      external_teams[0] > 1 || external_teams[1] > 1) {
    throw std::runtime_error("Invalid replay recording metadata");
  }
  stream_.write(kReplayMagic.data(), kReplayMagic.size());
  if (!stream_) throw std::runtime_error("Unable to write replay file");
  WriteValue(stream_, kReplayFormatVersion);
  for (std::uint8_t external : external_teams) WriteValue(stream_, external);
  WriteValue(stream_, match_duration);
  game_count_offset_ = stream_.tellp();
  WriteValue(stream_, std::int32_t{0});
  WriteValue(stream_, static_cast<std::uint64_t>(initial_state.size()));
  stream_.write(initial_state.data(),
                static_cast<std::streamsize>(initial_state.size()));
  if (!stream_) throw std::runtime_error("Unable to write replay state");
}

ReplayWriter::~ReplayWriter() = default;

// Starts a separately indexed match section in the replay stream.
void ReplayWriter::BeginGame() {
  if (finished_) throw std::runtime_error("Replay recording is already finished");
  if (game_active_) throw std::runtime_error("A replay game is already active");
  if (game_count_ >= kMaximumGameCount) {
    throw std::runtime_error("Replay contains too many games");
  }
  stream_.write(kGameMagic.data(), kGameMagic.size());
  if (!stream_) throw std::runtime_error("Unable to write replay game header");
  WriteValue(stream_, game_count_);
  current_step_count_offset_ = stream_.tellp();
  WriteValue(stream_, std::int32_t{0});
  current_final_step_offset_ = stream_.tellp();
  WriteValue(stream_, std::int32_t{0});
  current_goals_offset_ = stream_.tellp();
  WriteValue(stream_, std::int32_t{0});
  WriteValue(stream_, std::int32_t{0});
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
  for (int side = 0; side < 2; ++side) {
    if (step.has_observation[side] != external_teams_[side] ||
        step.has_decision[side] != external_teams_[side]) {
      throw std::runtime_error("Replay step does not match team types");
    }
  }
  WriteStep(stream_, step);
  ++current_step_count_;
  last_recorded_step_ = step.step;
}

// Finalizes the active match section with its terminal state.
void ReplayWriter::FinishGame(
    std::int32_t final_step, const std::array<std::int32_t, 2>& goals) {
  if (!game_active_) throw std::runtime_error("No replay game is active");
  if (current_step_count_ <= 0 || final_step != last_recorded_step_ + 1 ||
      final_step > match_duration_ || goals[0] < 0 || goals[1] < 0) {
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

// Finalizes file-level metadata after all recorded matches.
void ReplayWriter::Finish() {
  if (finished_) return;
  if (game_active_) throw std::runtime_error("Cannot finish an active replay game");
  const std::streampos end = stream_.tellp();
  stream_.seekp(game_count_offset_);
  WriteValue(stream_, game_count_);
  stream_.seekp(end);
  stream_.flush();
  if (!stream_) throw std::runtime_error("Unable to finalize replay file");
  finished_ = true;
}

// Loads and validates a complete replay into memory for deterministic playback.
ReplayReader::ReplayReader(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Unable to open replay file: " + path.string());
  }
  std::array<char, kReplayMagic.size()> magic{};
  stream.read(magic.data(), magic.size());
  if (!stream || magic != kReplayMagic) {
    throw std::runtime_error("Invalid gfootball replay file");
  }
  const std::uint32_t format_version = ReadValue<std::uint32_t>(stream);
  if (format_version != kReplayFormatVersion) {
    throw std::runtime_error("Unsupported gfootball replay format");
  }
  for (std::uint8_t& external : external_teams_) {
    external = ReadValue<std::uint8_t>(stream);
    if (external > 1) throw std::runtime_error("Invalid replay team type");
  }
  match_duration_ = ReadValue<std::int32_t>(stream);
  const std::int32_t game_count = ReadValue<std::int32_t>(stream);
  const std::uint64_t state_size = ReadValue<std::uint64_t>(stream);
  if (match_duration_ <= 0 || game_count <= 0 ||
      game_count > kMaximumGameCount || state_size == 0 ||
      state_size > kMaximumStateSize) {
    throw std::runtime_error("Invalid replay metadata");
  }
  initial_state_.resize(static_cast<std::size_t>(state_size));
  stream.read(initial_state_.data(), static_cast<std::streamsize>(state_size));
  if (!stream) throw std::runtime_error("Replay state is truncated");

  games_.reserve(static_cast<std::size_t>(game_count));
  for (std::int32_t game_index = 0; game_index < game_count; ++game_index) {
    std::array<char, kGameMagic.size()> game_magic{};
    stream.read(game_magic.data(), game_magic.size());
    if (!stream || game_magic != kGameMagic ||
        ReadValue<std::int32_t>(stream) != game_index) {
      throw std::runtime_error("Invalid replay game boundary");
    }
    const std::int32_t step_count = ReadValue<std::int32_t>(stream);
    ReplayGame game;
    game.final_step = ReadValue<std::int32_t>(stream);
    game.goals[0] = ReadValue<std::int32_t>(stream);
    game.goals[1] = ReadValue<std::int32_t>(stream);
    if (step_count <= 0 || step_count > kMaximumStepCount ||
        game.final_step <= 0 || game.final_step > match_duration_ ||
        game.goals[0] < 0 || game.goals[1] < 0) {
      throw std::runtime_error("Invalid replay game metadata");
    }
    game.steps.reserve(static_cast<std::size_t>(step_count));
    for (std::int32_t step_index = 0; step_index < step_count; ++step_index) {
      ReplayStep value = ReadStep(stream);
      if ((step_index == 0 && value.step != 0) ||
          (step_index > 0 && value.step != game.steps.back().step + 1)) {
        throw std::runtime_error("Replay decision steps are not contiguous");
      }
      for (int side = 0; side < 2; ++side) {
        if (value.has_observation[side] > 1 ||
            value.has_observation[side] != external_teams_[side] ||
            value.has_decision[side] > 1 ||
            value.has_decision[side] != external_teams_[side] ||
            (value.has_observation[side] &&
             value.observations[side].step != value.step) ||
            (value.has_decision[side] &&
             (value.decisions[side].controlled_player < 0 ||
              value.decisions[side].controlled_player >=
                  kGFootballPlayersPerTeam ||
              value.decisions[side].action < game_idle ||
              value.decisions[side].action > game_builtin_ai))) {
          throw std::runtime_error("Replay step does not match its header");
        }
      }
      game.steps.push_back(std::move(value));
    }
    if (game.final_step != game.steps.back().step + 1) {
      throw std::runtime_error("Replay game result does not match its steps");
    }
    games_.push_back(std::move(game));
  }
  if (stream.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("Replay file contains trailing data");
  }
}
