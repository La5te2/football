// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

#include <charconv>
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

namespace {

constexpr int kMatchDurationSteps = 3000;

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
  MessageBoxA(nullptr, message.c_str(), "gfootball", MB_OK | MB_ICONERROR);
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

bool IsBuiltin(const std::string& side) {
  return side == "builtin";
}

bool ParsePositiveInteger(const std::string& value, int& result) {
  result = 0;
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), result);
  return parsed.ec == std::errc{} &&
         parsed.ptr == value.data() + value.size() && result > 0;
}

std::filesystem::path ResolvePlugin(const std::string& value,
                                    const std::filesystem::path& root) {
  std::filesystem::path path = value;
  if (path.is_relative() && !std::filesystem::exists(path)) {
    path = root / path;
  }
  return path;
}

void LoadModels(GameEnv& env, const std::string& left,
                const std::string& right,
                const std::filesystem::path& root) {
  if (!IsBuiltin(left)) {
    env.load_model(ResolvePlugin(left, root), true, 0,
                   kMatchDurationSteps);
  }
  if (!IsBuiltin(right)) {
    env.load_model(ResolvePlugin(right, root), false, 0,
                   kMatchDurationSteps);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string left = "builtin";
  std::string right = "builtin";
  std::filesystem::path record_path;
  int games = 1;
  bool render = true;
  bool real_time = true;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--left" && i + 1 < argc) {
      left = argv[++i];
    } else if (argument == "--right" && i + 1 < argc) {
      right = argv[++i];
    } else if (argument.rfind("--left=", 0) == 0) {
      left = argument.substr(7);
    } else if (argument.rfind("--right=", 0) == 0) {
      right = argument.substr(8);
    } else if (argument == "--record" && i + 1 < argc) {
      record_path = argv[++i];
    } else if (argument.rfind("--record=", 0) == 0) {
      record_path = argument.substr(9);
    } else if (argument == "--games" && i + 1 < argc) {
      if (!ParsePositiveInteger(argv[++i], games)) {
        ShowError("--games requires a positive integer");
        return 2;
      }
    } else if (argument.rfind("--games=", 0) == 0) {
      if (!ParsePositiveInteger(argument.substr(8), games)) {
        ShowError("--games requires a positive integer");
        return 2;
      }
    } else if (argument == "--render=false") {
      render = false;
    } else if (argument == "--real_time=false") {
      real_time = false;
    } else {
      ShowError("Unknown option: " + argument);
      return 2;
    }
  }

  const std::filesystem::path executable =
      std::filesystem::absolute(argv[0]).parent_path();
  const std::filesystem::path root =
      std::filesystem::exists(executable / "models")
          ? executable
          : executable.parent_path();
  SetEnvironment("GFOOTBALL_DATA_DIR", (executable / "data").string());
  SetEnvironment("GFOOTBALL_FONT",
                 (executable / "fonts" /
                  "AlegreyaSansSC-ExtraBold.ttf").string());
  SetDefaultEnvironment("MESA_GL_VERSION_OVERRIDE", "3.2");
  SetDefaultEnvironment("MESA_GLSL_VERSION_OVERRIDE", "150");

  try {
    const int left_agents = IsBuiltin(left) ? 0 : 1;
    const int right_agents = IsBuiltin(right) ? 0 : 1;
    auto simulation = std::make_unique<GameEnv>();
    simulation->game_config.physics_steps_per_frame = 10;
    simulation->start_game();
    auto simulation_config =
        MakeScenario(left_agents, right_agents, render && real_time);
    simulation->state = game_running;
    simulation->reset(*simulation_config, false);

    GameEnv* env = simulation.get();
    std::unique_ptr<GameEnv> rendering;
    if (render) {
      while (true) {
        SharedInfo state;
        {
          ContextHolder context(simulation.get());
          state = simulation->get_info();
        }
        if (state.is_in_play) break;
        {
          ContextHolder context(simulation.get());
          simulation->step();
        }
      }
      const std::string state = simulation->get_state("");
      rendering = std::make_unique<GameEnv>();
      rendering->game_config.render = true;
      rendering->game_config.physics_steps_per_frame = 10;
      rendering->start_game();
      auto rendering_config =
          MakeScenario(left_agents, right_agents, real_time);
      rendering->state = game_running;
      rendering->game_config.render_frames_per_step = 6;
      rendering->reset(*rendering_config, false);
      {
        ContextHolder context(rendering.get());
        rendering->set_state(state);
        rendering->get_info();
        rendering->get_frame();
        rendering->render(false);
        rendering->render(true);
      }
      env = rendering.get();
    }

    LoadModels(*env, left, right, root);
    const std::string initial_state = env->get_state("");
    if (!record_path.empty()) {
      env->start_recording(record_path, initial_state,
                           !IsBuiltin(left), !IsBuiltin(right),
                           kMatchDurationSteps);
    }
    for (int game = 0; game < games && !env->window_closed(); ++game) {
      if (game > 0) {
        {
          ContextHolder context(env);
          env->set_state(initial_state);
        }
        env->reset_models();
      }
      env->begin_recording_game();
      SharedInfo final_state;
      while (!env->window_closed()) {
        {
          ContextHolder context(env);
          env->step();
          final_state = env->get_info();
        }
        if (final_state.step >= kMatchDurationSteps) break;
      }
      env->finish_recording_game(final_state);
    }
    if (!record_path.empty()) {
      env->finish_recording();
    }
  } catch (const std::exception& error) {
    ShowError(error.what());
    return 1;
  }
  return 0;
}
