// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

// Adapter between the TamakEri policy and the common model plugin interface.

#include "adapter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {

enum EngineAction {
  kIdle = 0,
  kLeft = 1,
  kTopLeft = 2,
  kTop = 3,
  kTopRight = 4,
  kRight = 5,
  kBottomRight = 6,
  kBottom = 7,
  kBottomLeft = 8,
  kLongPass = 9,
  kHighPass = 10,
  kShortPass = 11,
  kShot = 12,
  kSliding = 14,
  kSprint = 18,
  kDribble = 19,
  kReleaseDirection = 20,
  kReleaseSprint = 30,
  kReleaseDribble = 31,
};

constexpr int kPolicyActionCount = 52;
constexpr int kBuiltinAiPolicyAction = 19;
constexpr int kDirectedKickBegin = 20;

float Distance(float ax, float ay, float bx, float by) {
  const float dx = ax - bx;
  const float dy = ay - by;
  return std::sqrt(dx * dx + dy * dy);
}

float DistanceMeters(float ax, float ay, float bx, float by) {
  constexpr float kHalfPitchLength = 54.4f;
  constexpr float kHalfPitchWidth = 83.6f;
  return Distance((ax - bx) * kHalfPitchLength,
                  (ay - by) * kHalfPitchWidth, 0.0f, 0.0f);
}

// Approximates arrival time from public motion state for TamakEri's local player
// selection without using an engine-provided tactical value.
int EstimateInterceptionTime(const GFootballModelPlayer& player,
                             const GFootballModelObservation& state) {
  constexpr float kReachPerStep = 0.8f;
  for (int step = 0; step <= 30; ++step) {
    const float ball_x =
        state.ball_position[0] + state.ball_direction[0] * step;
    const float ball_y =
        state.ball_position[1] + state.ball_direction[1] * step;
    const int momentum_steps = std::min(step, 3);
    const float player_x =
        player.position[0] + player.direction[0] * momentum_steps;
    const float player_y =
        player.position[1] + player.direction[1] * momentum_steps;
    if (DistanceMeters(player_x, player_y, ball_x, ball_y) <=
        0.5f + kReachPerStep * step) {
      return step * 100;
    }
  }
  return 3000 + static_cast<int>(100 * DistanceMeters(
      player.position[0], player.position[1], state.ball_position[0],
      state.ball_position[1]));
}

float MultiScale(float value, float scale) {
  return 2.0f / (1.0f + std::exp(-value / scale));
}

at::Tensor FloatTensor(const std::vector<float>& values,
                       std::initializer_list<int64_t> shape) {
  return torch::from_blob(const_cast<float*>(values.data()), shape,
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

at::Tensor LongTensor(const std::vector<int64_t>& values,
                      std::initializer_list<int64_t> shape) {
  return torch::from_blob(const_cast<int64_t*>(values.data()), shape,
                          torch::TensorOptions().dtype(torch::kInt64))
      .clone();
}

int OppositeDirection(int action) {
  static constexpr int directions[] = {0, 5, 6, 7, 8, 1, 2, 3, 4};
  return action >= 1 && action <= 8 ? directions[action] : action;
}

// Translates the policy's 19 directly executable atomic actions from TamakEri
// indices into the distinct engine Action namespace.
int DirectedToBackend(int action) {
  static constexpr int actions[] = {
      kIdle, kLeft, kTopLeft, kTop, kTopRight,
      kRight, kBottomRight, kBottom, kBottomLeft,
      kLongPass, kHighPass, kShortPass, kShot,
      kSprint, kReleaseDirection, kReleaseSprint,
      kSliding, kDribble, kReleaseDribble};
  return actions[action];
}

}  // namespace

// Loads the published TorchScript policy and prepares it for inference.
TamakEriAdapter::TamakEriAdapter(const std::string& model_path, int side,
                                 int game_duration)
    : model_(torch::jit::load(model_path)),
      side_(side),
      game_duration_(game_duration) {
  model_.eval();
  Reset();
}

// Clears player selection, delayed actions, and policy history for a new match.
void TamakEriAdapter::Reset() {
  selected_player_ = -1;
  pending_action_ = -1;
  last_action_ = 0;
  action_history_.clear();
}

// Reconstructs TamakEri's engine-like possession selector locally from public
// player and ball state.
void TamakEriAdapter::SelectPlayer(
    const GFootballModelObservation& observation) {
  const auto& team = observation.teams[side_];
  int candidate = -1;
  if (observation.ball_owned_team == side_ &&
      observation.ball_owned_player > 0 &&
      observation.ball_owned_player < kGFootballPlayersPerTeam &&
      team[observation.ball_owned_player].is_active) {
    candidate = observation.ball_owned_player;
  }

  int candidate_time = std::numeric_limits<int>::max();
  if (candidate < 0) {
    for (int i = 0; i < kGFootballPlayersPerTeam; ++i) {
      if (!team[i].is_active || team[i].role == 0) continue;
      const int time = EstimateInterceptionTime(team[i], observation);
      if (time < candidate_time) {
        candidate_time = time;
        candidate = i;
      }
    }
  } else {
    candidate_time = EstimateInterceptionTime(team[candidate], observation);
  }

  if (candidate < 0) {
    for (int i = 0; i < kGFootballPlayersPerTeam; ++i) {
      if (team[i].is_active) {
        candidate = i;
        candidate_time = EstimateInterceptionTime(team[i], observation);
        break;
      }
    }
  }

  if (selected_player_ < 0 || selected_player_ >= kGFootballPlayersPerTeam ||
      !team[selected_player_].is_active) {
    selected_player_ = candidate;
  } else if (candidate >= 0 && candidate != selected_player_) {
    const int selected_time =
        EstimateInterceptionTime(team[selected_player_], observation);
    const bool candidate_has_ball =
        observation.ball_owned_team == side_ &&
        observation.ball_owned_player == candidate;
    if (candidate_has_ball || observation.game_mode != 0 ||
        3 * candidate_time < selected_time) {
      selected_player_ = candidate;
    }
  }
}

// Canonicalizes either physical side into TamakEri's own-team-attacks-right
// coordinate system and reproduces the compact observation used in training.
TamakEriAdapter::Observation TamakEriAdapter::Convert(
    const GFootballModelObservation& state) const {
  Observation result;
  const auto& own = state.teams[side_];
  const auto& opponent = state.teams[1 - side_];
  const float rotation = side_ == 0 ? 1.0f : -1.0f;
  result.ball = {
      rotation * state.ball_position[0], rotation * state.ball_position[1],
      state.ball_position[2], rotation * state.ball_direction[0],
      rotation * state.ball_direction[1], state.ball_direction[2],
      state.ball_rotation[0], state.ball_rotation[1], state.ball_rotation[2]};
  for (int i = 0; i < 11; ++i) {
    result.own_position[i] = {
        rotation * own[i].position[0], rotation * own[i].position[1]};
    result.opponent_position[i] = {
        rotation * opponent[i].position[0],
        rotation * opponent[i].position[1]};
    result.own_direction[i] = {
        rotation * own[i].direction[0], rotation * own[i].direction[1]};
    result.opponent_direction[i] = {
        rotation * opponent[i].direction[0],
        rotation * opponent[i].direction[1]};
    result.own_tired[i] = own[i].tired_factor;
    result.opponent_tired[i] = opponent[i].tired_factor;
    result.own_card[i] = own[i].has_card ? 1.0f : 0.0f;
    result.opponent_card[i] = opponent[i].has_card ? 1.0f : 0.0f;
    result.own_active[i] = own[i].is_active ? 1.0f : 0.0f;
    result.opponent_active[i] = opponent[i].is_active ? 1.0f : 0.0f;
  }
  for (int direction = 1; direction <= 8; ++direction) {
    const int actual = side_ == 0 ? direction : OppositeDirection(direction);
    result.sticky[direction - 1] = state.sticky_actions[actual - 1] ? 1 : 0;
  }
  result.sticky[8] = state.sticky_actions[8] ? 1 : 0;
  result.sticky[9] = state.sticky_actions[9] ? 1 : 0;
  result.score = side_ == 0
      ? std::array<int, 2>{state.goals[0], state.goals[1]}
      : std::array<int, 2>{state.goals[1], state.goals[0]};
  result.controlled_player = selected_player_;
  result.ball_owned_team = state.ball_owned_team < 0
      ? -1
      : (side_ == 0 ? state.ball_owned_team : 1 - state.ball_owned_team);
  result.ball_owned_player = state.ball_owned_player;
  result.game_mode = state.game_mode;
  result.steps_left = game_duration_ - state.step;

  if (result.game_mode == 2 || result.game_mode == 3 ||
      result.game_mode == 4 || result.game_mode == 6) {
    float best = std::numeric_limits<float>::max();
    for (int side = 0; side < 2; ++side) {
      const auto& positions = side == 0
          ? result.own_position : result.opponent_position;
      for (int i = 0; i < 11; ++i) {
        const float distance = Distance(
            positions[i][0], positions[i][1], result.ball[0], result.ball[1]);
        if (distance < best) {
          best = distance;
          result.ball_owned_team = side;
          result.ball_owned_player = i;
        }
      }
    }
  }
  return result;
}

// Preserves the tensor order, shapes, scaling, and action history required by
// the published TamakEri weights.
std::vector<torch::jit::IValue> TamakEriAdapter::Features(
    const Observation& observation) const {
  std::vector<float> ball(observation.ball.begin(), observation.ball.end());
  std::vector<float> match;
  for (int score : observation.score) {
    match.push_back(MultiScale(score, 1.0f));
    match.push_back(MultiScale(score, 3.0f));
  }
  const int difference = observation.score[0] - observation.score[1];
  match.push_back(MultiScale(difference, 1.0f));
  match.push_back(MultiScale(difference, 3.0f));
  for (float scale : {10.0f, 100.0f, 1000.0f, 10000.0f}) {
    match.push_back(MultiScale(observation.steps_left, scale));
  }
  int half_left = observation.steps_left;
  if (half_left > 1500) half_left -= 1500;
  for (float scale : {10.0f, 100.0f, 1000.0f, 10000.0f}) {
    match.push_back(MultiScale(half_left, scale));
  }
  match.push_back(observation.ball_owned_team == 0 ? 1.0f : 0.0f);
  match.push_back(observation.ball_owned_team == 1 ? 1.0f : 0.0f);

  std::vector<float> own_players;
  std::vector<float> opponent_players;
  for (int side = 0; side < 2; ++side) {
    auto& output = side == 0 ? own_players : opponent_players;
    const auto& position = side == 0
        ? observation.own_position : observation.opponent_position;
    const auto& direction = side == 0
        ? observation.own_direction : observation.opponent_direction;
    const auto& tired = side == 0
        ? observation.own_tired : observation.opponent_tired;
    const auto& card = side == 0
        ? observation.own_card : observation.opponent_card;
    const auto& active = side == 0
        ? observation.own_active : observation.opponent_active;
    for (int i = 0; i < 11; ++i) {
      output.insert(output.end(), {
          side == 0 ? 1.0f : 0.0f,
          position[i][0], position[i][1], direction[i][0], direction[i][1],
          tired[i], card[i], active[i],
          observation.ball_owned_team == side &&
                  observation.ball_owned_player == i ? 1.0f : 0.0f});
    }
  }

  static constexpr float direction_vectors[8][2] = {
      {-1.0f, 0.0f}, {-0.707f, -0.707f}, {0.0f, 1.0f},
      {0.707f, -0.707f}, {1.0f, 0.0f}, {0.707f, 0.707f},
      {0.0f, -1.0f}, {-0.707f, 0.707f}};
  std::vector<float> control = {0.0f, 0.0f,
                                static_cast<float>(observation.sticky[8]),
                                static_cast<float>(observation.sticky[9])};
  for (int i = 0; i < 8; ++i) {
    if (observation.sticky[i]) {
      control[0] = direction_vectors[i][0];
      control[1] = direction_vectors[i][1];
      break;
    }
  }
  std::vector<int64_t> indices(11);
  for (int i = 0; i < 11; ++i) indices[i] = i;
  std::vector<int64_t> mode = {observation.game_mode};
  std::vector<float> control_flag(22, 0.0f);
  if (observation.controlled_player >= 0 &&
      observation.controlled_player < 11) {
    control_flag[observation.controlled_player] = 1.0f;
  }

  std::vector<float> player_distances;
  for (const auto* positions : {&observation.own_position,
                                &observation.opponent_position}) {
    for (const auto& position : *positions) {
      player_distances.insert(player_distances.end(), {
          Distance(position[0], position[1], observation.ball[0], observation.ball[1]),
          Distance(position[0], position[1], -1.0f, 0.0f),
          Distance(position[0], position[1], 1.0f, 0.0f),
          std::abs(position[0] + 1.0f), std::abs(position[0] - 1.0f),
          std::abs(position[1] + 0.42f), std::abs(position[1] - 0.42f)});
    }
  }
  std::vector<float> ball_distances = {
      Distance(observation.ball[0], observation.ball[1], -1.0f, 0.0f),
      Distance(observation.ball[0], observation.ball[1], 1.0f, 0.0f),
      std::abs(observation.ball[0] + 1.0f),
      std::abs(observation.ball[0] - 1.0f),
      std::abs(observation.ball[1] + 0.42f),
      std::abs(observation.ball[1] - 0.42f)};

  std::vector<float> grid(53 * 11 * 11);
  const int active = std::max(0, observation.controlled_player);
  for (int i = 0; i < 11; ++i) {
    for (int j = 0; j < 11; ++j) {
      const auto& left = observation.own_position[i];
      const auto& right = observation.opponent_position[j];
      const auto& left_d = observation.own_direction[i];
      const auto& right_d = observation.opponent_direction[j];
      const float values[53] = {
          left[0], left[1], right[0], right[1],
          observation.ball[0], observation.ball[1], observation.ball[2],
          -1.0f, 0.0f, 1.0f, 0.0f, -0.42f, 0.42f,
          observation.own_position[active][0],
          observation.own_position[active][1],
          left[0] - right[0], left[1] - right[1],
          left[0] - 1.0f, left[1], left[0] + 1.0f, left[1],
          right[0] - 1.0f, right[1], right[0] + 1.0f, right[1],
          std::abs(left[1] + 0.42f), std::abs(left[1] - 0.42f),
          std::abs(right[1] + 0.42f), std::abs(right[1] - 0.42f),
          right[0] - observation.ball[0], right[1] - observation.ball[1],
          right[0] - observation.own_position[active][0],
          right[1] - observation.own_position[active][1],
          left[0] - observation.ball[0], left[1] - observation.ball[1],
          left[0] - observation.own_position[active][0],
          left[1] - observation.own_position[active][1],
          observation.ball[3], observation.ball[4], observation.ball[5],
          left_d[0] - observation.ball[3], left_d[1] - observation.ball[4],
          right_d[0] - observation.ball[3], right_d[1] - observation.ball[4],
          left_d[0], left_d[1], right_d[0], right_d[1],
          left_d[0] - right_d[0], left_d[1] - right_d[1],
          observation.ball[6], observation.ball[7], observation.ball[8]};
      for (int channel = 0; channel < 53; ++channel) {
        grid[(channel * 11 + i) * 11 + j] = values[channel];
      }
    }
  }

  std::vector<int64_t> history;
  for (auto it = action_history_.rbegin(); it != action_history_.rend(); ++it) {
    history.push_back(*it);
  }
  while (history.size() < 8) history.push_back(0);
  history.resize(8);

  return {
      FloatTensor(ball, {1, 9}), FloatTensor(match, {1, 16}),
      FloatTensor(own_players, {1, 11, 9}),
      FloatTensor(opponent_players, {1, 11, 9}),
      FloatTensor(control, {1, 4}), LongTensor(indices, {1, 11}),
      LongTensor(indices, {1, 11}), LongTensor(mode, {1, 1}),
      FloatTensor(control_flag, {1, 22, 1}),
      FloatTensor(player_distances, {1, 22, 7}),
      FloatTensor(ball_distances, {1, 6}), FloatTensor(grid, {1, 53, 11, 11}),
      LongTensor(history, {1, 8, 1})};
}

// Applies TamakEri's state-dependent legal-action mask before taking argmax.
int TamakEriAdapter::ChooseAction(const Observation& observation,
                                  const at::Tensor& logits) const {
  float best_value = -std::numeric_limits<float>::infinity();
  int best_action = 0;
  const auto values = logits.contiguous().view({-1});
  const auto accessor = values.accessor<float, 1>();
  for (int action = 0; action < kPolicyActionCount; ++action) {
    // Policy index 19 is game_builtin_ai. TamakEri excludes it so the external
    // model remains the sole controller of its selected player instead of
    // sharing control with ElizaController. Player selection is independent.
    bool legal = action != kBuiltinAiPolicyAction;
    const bool owns_ball = observation.ball_owned_team == 0;
    if (!owns_ball && ((action >= 9 && action <= 12) || action == 17 ||
                       action >= kDirectedKickBegin)) legal = false;
    if (owns_ball && action == 16) legal = false;
    if (!observation.sticky[8] && action == 15) legal = false;
    if (!observation.sticky[9] && action == 18) legal = false;
    if (std::none_of(observation.sticky.begin(),
                     observation.sticky.begin() + 8,
                     [](int value) { return value != 0; }) && action == 14) {
      legal = false;
    }
    if (legal && accessor[action] > best_value) {
      best_value = accessor[action];
      best_action = action;
    }
  }
  return best_action;
}

// Translates a policy action and mirrors directional input for the physical
// right team.
int TamakEriAdapter::Submit(int action) {
  int backend = DirectedToBackend(action);
  if (side_ != 0 && backend >= kLeft && backend <= kBottomLeft) {
    backend = OppositeDirection(backend);
  }
  last_action_ = action;
  return backend;
}

// Chooses the recipient, updates policy history, runs one inference, and splits
// directed kicks into a kick followed by a direction.
GFootballModelDecision TamakEriAdapter::Decide(
    const GFootballModelObservation& state) {
  SelectPlayer(state);
  if (selected_player_ < 0) return {-1, kIdle};
  action_history_.push_back(last_action_);
  while (action_history_.size() > 8) action_history_.pop_front();
  if (pending_action_ >= 0) {
    const int pending = pending_action_;
    pending_action_ = -1;
    return {selected_player_, Submit(pending)};
  }
  const Observation observation = Convert(state);
  torch::InferenceMode inference_mode;
  const at::Tensor logits = model_.forward(Features(observation)).toTensor();
  const int action = ChooseAction(observation, logits);
  if (action >= kDirectedKickBegin) {
    const int offset = action - kDirectedKickBegin;
    const int kick = offset / 8;
    pending_action_ = offset % 8 + 1;
    return {selected_player_, Submit(9 + kick)};
  }
  return {selected_player_, Submit(action)};
}
