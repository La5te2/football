// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

// Standalone renderer for native gfootball replay files.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#endif

#include "game_env.hpp"
#include "replay.hpp"

namespace {

void SetEnvironment(const char* name, const std::string& value) {
#ifdef _WIN32
  _putenv_s(name, value.c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

void SetDefaultEnvironment(const char* name, const std::string& value) {
  if (!std::getenv(name)) SetEnvironment(name, value);
}

void ShowError(const std::string& message) {
#ifdef _WIN32
  MessageBoxA(nullptr, message.c_str(), "gfootball replay",
              MB_OK | MB_ICONERROR);
#else
  std::cerr << message << '\n';
#endif
}

void AddFormation(std::vector<FormationEntry>& team, bool kickoff_team) {
  struct Entry {
    float x;
    float y;
    e_PlayerRole role;
  };
  static constexpr Entry formation[] = {
      {-1.0f, 0.0f, e_PlayerRole_GK},
      {0.0f, 0.02f, e_PlayerRole_RM},
      {0.0f, -0.02f, e_PlayerRole_CF},
      {-0.422f, -0.19576f, e_PlayerRole_LB},
      {-0.5f, -0.06356f, e_PlayerRole_CB},
      {-0.5f, 0.063559f, e_PlayerRole_CB},
      {-0.422f, 0.19576f, e_PlayerRole_RB},
      {-0.184212f, -0.10568f, e_PlayerRole_CM},
      {-0.267574f, 0.0f, e_PlayerRole_CM},
      {-0.184212f, 0.10568f, e_PlayerRole_CM},
      {-0.01f, -0.2161f, e_PlayerRole_LM},
  };
  for (int i = 0; i < 11; ++i) {
    float x = formation[i].x;
    float y = formation[i].y;
    if (!kickoff_team && i == 1) {
      x = -0.05f;
      y = 0.0f;
    } else if (!kickoff_team && i == 2) {
      x = -0.01f;
      y = 0.216102f;
    }
    team.emplace_back(x, y, formation[i].role, false, true);
  }
}

boost::shared_ptr<ScenarioConfig> MakeScenario(
    int left_agents, int right_agents, bool real_time) {
  auto config = ScenarioConfig::make();
  config->left_agents = left_agents;
  config->right_agents = right_agents;
  config->real_time = real_time;
  AddFormation(config->left_team, true);
  AddFormation(config->right_team, false);
  return config;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path replay_path;
  bool render = true;
  bool real_time = true;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--render=false") {
      render = false;
    } else if (argument == "--real_time=false") {
      real_time = false;
    } else if (!argument.empty() && argument[0] != '-' &&
               replay_path.empty()) {
      replay_path = argument;
    } else {
      ShowError("Unknown option: " + argument);
      return 2;
    }
  }
  if (replay_path.empty()) {
    ShowError("Usage: replay.exe <replay-file>");
    return 2;
  }

  const std::filesystem::path executable =
      std::filesystem::absolute(argv[0]).parent_path();
  SetEnvironment("GFOOTBALL_DATA_DIR", (executable / "data").string());
  SetEnvironment("GFOOTBALL_FONT",
                 (executable / "fonts" /
                  "AlegreyaSansSC-ExtraBold.ttf").string());
  SetDefaultEnvironment("MESA_GL_VERSION_OVERRIDE", "3.2");
  SetDefaultEnvironment("MESA_GLSL_VERSION_OVERRIDE", "150");

  try {
    ReplayReader replay(replay_path);
    const int left_agents = replay.external_team(0) ? 1 : 0;
    const int right_agents = replay.external_team(1) ? 1 : 0;

    auto env = std::make_unique<GameEnv>();
    env->game_config.render = render;
    env->game_config.physics_steps_per_frame = 10;
    env->start_game();
    auto config = MakeScenario(left_agents, right_agents, real_time);
    env->state = game_running;
    if (render) env->game_config.render_frames_per_step = 6;
    env->reset(*config, false);
    {
      ContextHolder context(env.get());
      env->set_state(replay.initial_state());
      env->scenario_config.real_time = real_time;
      env->get_info();
      if (render) {
        env->get_frame();
        env->render(false);
        env->render(true);
      }
    }

    for (std::size_t game_index = 0;
         game_index < replay.game_count() && !env->window_closed();
         ++game_index) {
      if (game_index > 0) {
        ContextHolder context(env.get());
        env->set_state(replay.initial_state());
        env->scenario_config.real_time = real_time;
        if (render) {
          env->get_frame();
          env->render(false);
          env->render(true);
        }
      }
      const ReplayGame& game = replay.game(game_index);
      env->start_replay(game);
      SharedInfo state;
      {
        ContextHolder context(env.get());
        state = env->get_info();
      }
      while (!env->window_closed() && state.step < game.final_step) {
        ContextHolder context(env.get());
        env->step();
        state = env->get_info();
      }
      if (!env->window_closed() &&
          (!env->replay_complete() || state.step != game.final_step ||
           state.left_goals != game.goals[0] ||
           state.right_goals != game.goals[1])) {
        throw std::runtime_error(
            "Replay game result does not match its recording");
      }
    }
  } catch (const std::exception& error) {
    ShowError(error.what());
    return 1;
  }
  return 0;
}
