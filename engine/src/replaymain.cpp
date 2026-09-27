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

void ShowError(const std::string& message, bool dialog) {
#ifdef _WIN32
  if (dialog) {
    MessageBoxA(nullptr, message.c_str(), "gfootball replay",
                MB_OK | MB_ICONERROR);
    return;
  }
#endif
  std::cerr << message << '\n';
}

// Reads the scenario already stored in an engine state so the replay
// environment can be constructed before that state is restored.
boost::shared_ptr<ScenarioConfig> ReadScenario(
    GameEnv& env, const std::string& state) {
  EnvState reader(&env, state);
  std::string pickle;
  reader.process(pickle);
  GameState game_state = game_created;
  reader.process(game_state);
  char random_state_byte = 0;
  for (std::size_t index = 0; index < sizeof(env.context->rng); ++index) {
    reader.process(random_state_byte);
  }
  auto recorded_config = ScenarioConfig::make();
  recorded_config->ProcessStateConstant(&reader);
  recorded_config->ProcessState(&reader);
  if (recorded_config->left_agents < 0 ||
      recorded_config->left_agents > kGFootballPlayersPerTeam ||
      recorded_config->right_agents < 0 ||
      recorded_config->right_agents > kGFootballPlayersPerTeam) {
    throw std::runtime_error("Replay contains invalid controller counts");
  }
  return recorded_config;
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
      ShowError("Unknown option: " + argument, render);
      return 2;
    }
  }
  if (replay_path.empty()) {
    ShowError("Usage: replay.exe <replay-file>", render);
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

    auto env = std::make_unique<GameEnv>();
    env->game_config.render = render;
    env->game_config.physics_steps_per_frame = 10;
    env->start_game();
    auto config = ReadScenario(*env, replay.game(0).initial_state);
    config->real_time = real_time;
    env->state = game_running;
    if (render) env->game_config.render_frames_per_step = 6;
    env->reset(*config, false);
    env->scenario_config =
        *ReadScenario(*env, replay.game(0).initial_state);

    for (std::size_t game_index = 0;
         game_index < replay.game_count() && !env->window_closed();
         ++game_index) {
      const ReplayGame& game = replay.game(game_index);
      {
        ContextHolder context(env.get());
        env->set_random_seed(game.seed);
        env->set_state(game.initial_state);
        if (env->scenario_config.game_engine_random_seed != game.seed) {
          throw std::runtime_error(
              "Replay seed does not match its initial state");
        }
        env->scenario_config.real_time = real_time;
        env->get_info();
        if (render) {
          env->get_frame();
          env->render(false);
          env->render(true);
        }
      }
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
    ShowError(error.what(), render);
    return 1;
  }
  return 0;
}
