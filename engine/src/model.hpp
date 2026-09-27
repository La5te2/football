// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

#ifndef GFOOTBALL_MODEL_HPP
#define GFOOTBALL_MODEL_HPP

#include <filesystem>

#include "interface.hpp"

class GameEnv;
struct SharedInfo;

// Engine-private owner and dispatcher for one external model DLL.
class Model final {
 public:
  Model(const std::filesystem::path& path, bool left_team, int game_duration);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  void Reset();
  void Decide(GameEnv& env, const SharedInfo& state);

 private:
  void* library_ = nullptr;
  GFootballModelHandle model_ = nullptr;
  GFootballModelDestroy destroy_ = nullptr;
  GFootballModelReset reset_ = nullptr;
  GFootballModelDecide decide_ = nullptr;
  bool left_team_ = false;
};

#endif
