// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

#ifndef GFOOTBALL_TAMAKERI_ADAPTER_HPP
#define GFOOTBALL_TAMAKERI_ADAPTER_HPP

#include <array>
#include <deque>
#include <string>
#include <vector>

#include <torch/script.h>

#include "interface.hpp"

// Implements every TamakEri-specific concern behind the model-neutral ABI:
// local player selection, feature construction, policy decoding, and LibTorch
// inference. It derives player selection only from public match observations.
class TamakEriAdapter {
 public:
  TamakEriAdapter(const std::string& model_path, int side,
                  int game_duration = 3000);
  void Reset();
  GFootballModelDecision Decide(
      const GFootballModelObservation& observation);

 private:
  struct Observation {
    std::array<float, 9> ball{};
    std::array<std::array<float, 2>, 11> own_position{};
    std::array<std::array<float, 2>, 11> opponent_position{};
    std::array<std::array<float, 2>, 11> own_direction{};
    std::array<std::array<float, 2>, 11> opponent_direction{};
    std::array<float, 11> own_tired{};
    std::array<float, 11> opponent_tired{};
    std::array<float, 11> own_card{};
    std::array<float, 11> opponent_card{};
    std::array<float, 11> own_active{};
    std::array<float, 11> opponent_active{};
    std::array<int, 10> sticky{};
    std::array<int, 2> score{};
    int controlled_player = -1;
    int ball_owned_team = -1;
    int ball_owned_player = -1;
    int game_mode = 0;
    int steps_left = 0;
  };

  void SelectPlayer(const GFootballModelObservation& observation);
  Observation Convert(const GFootballModelObservation& observation) const;
  std::vector<torch::jit::IValue> Features(const Observation& observation) const;
  int ChooseAction(const Observation& observation,
                   const at::Tensor& logits) const;
  int Submit(int action);

  torch::jit::script::Module model_;
  int side_;
  int game_duration_;
  int selected_player_ = -1;
  int pending_action_ = -1;
  int last_action_ = 0;
  std::deque<int> action_history_;
};

#endif
