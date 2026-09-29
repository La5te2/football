// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

// CPython bindings for the native reinforcement-learning environment.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "gfootball_actions.h"
#include "environment.hpp"

namespace {

constexpr char kCapsuleName[] = "gfootball.TrainingEnvironment";
constexpr char kBatchCapsuleName[] = "gfootball.TrainingEnvironmentBatch";

class TrainingEnvironmentBatch {
 public:
  TrainingEnvironmentBatch(const std::filesystem::path& data_directory,
                           const std::filesystem::path& font_file,
                           int maximum_steps, int count)
      : maximum_steps_(maximum_steps) {
    if (count <= 0) {
      throw std::invalid_argument("environment count must be positive");
    }
    environments_.reserve(count);
    for (int index = 0; index < count; ++index) {
      environments_.push_back(std::make_unique<TrainingEnvironment>(
          data_directory, font_file, maximum_steps));
    }
    actions_.resize(count);
    observations_.resize(count);
    workers_.reserve(count);
    for (int index = 0; index < count; ++index) {
      workers_.emplace_back(&TrainingEnvironmentBatch::Worker, this, index);
    }
  }

  ~TrainingEnvironmentBatch() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      ++generation_;
    }
    work_ready_.notify_all();
    for (std::thread& worker : workers_) worker.join();
  }

  int size() const { return static_cast<int>(environments_.size()); }
  int maximum_steps() const { return maximum_steps_; }

  std::vector<TrainingObservation> Reset(
      const std::vector<std::uint32_t>& seeds,
      const std::vector<bool>& left_teams) {
    if (seeds.size() != environments_.size() ||
        left_teams.size() != environments_.size()) {
      throw std::invalid_argument(
          "batch reset requires one seed and side per environment");
    }
    // Engine reset seeds a legacy process-wide C RNG before seeding the
    // match-local stream, so resets remain serial and deterministic.
    for (std::size_t index = 0; index < environments_.size(); ++index) {
      observations_[index] =
          environments_[index]->Reset(seeds[index], left_teams[index]);
    }
    return observations_;
  }

  std::vector<TrainingObservation> ResetBuiltin(
      const std::vector<std::uint32_t>& seeds,
      const std::vector<bool>& left_teams) {
    if (seeds.size() != environments_.size() ||
        left_teams.size() != environments_.size()) {
      throw std::invalid_argument(
          "batch reset requires one seed and side per environment");
    }
    for (std::size_t index = 0; index < environments_.size(); ++index) {
      observations_[index] = environments_[index]->ResetBuiltin(
          seeds[index], left_teams[index]);
    }
    return observations_;
  }

  TrainingObservation ResetOne(int index, std::uint32_t seed,
                               bool left_team) {
    if (index < 0 || index >= size()) {
      throw std::out_of_range("environment index is out of range");
    }
    observations_[index] = environments_[index]->Reset(seed, left_team);
    return observations_[index];
  }

  std::vector<TrainingObservation> Step(
      const std::vector<std::array<std::int32_t,
                                   kGFootballPlayersPerTeam>>& actions) {
    if (actions.size() != environments_.size()) {
      throw std::invalid_argument(
          "batch step requires one decision per environment");
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      actions_ = actions;
      completed_ = 0;
      exception_ = nullptr;
      ++generation_;
    }
    work_ready_.notify_all();
    std::unique_lock<std::mutex> lock(mutex_);
    work_complete_.wait(
        lock, [&]() { return completed_ == environments_.size(); });
    if (exception_) std::rethrow_exception(exception_);
    return observations_;
  }

 private:
  void Worker(int index) {
    std::size_t observed_generation = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      work_ready_.wait(lock, [&]() {
        return stopping_ || generation_ != observed_generation;
      });
      if (stopping_) return;
      observed_generation = generation_;
      lock.unlock();
      try {
        observations_[index] =
            environments_[index]->Step(actions_[index]);
      } catch (...) {
        lock.lock();
        if (!exception_) exception_ = std::current_exception();
        lock.unlock();
      }
      lock.lock();
      ++completed_;
      const bool finished = completed_ == environments_.size();
      lock.unlock();
      if (finished) work_complete_.notify_one();
    }
  }

  int maximum_steps_;
  std::vector<std::unique_ptr<TrainingEnvironment>> environments_;
  std::vector<std::array<std::int32_t, kGFootballPlayersPerTeam>> actions_;
  std::vector<TrainingObservation> observations_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable work_ready_;
  std::condition_variable work_complete_;
  std::size_t generation_ = 0;
  std::size_t completed_ = 0;
  bool stopping_ = false;
  std::exception_ptr exception_;
};

TrainingEnvironment* GetEnvironment(PyObject* capsule) {
  return static_cast<TrainingEnvironment*>(
      PyCapsule_GetPointer(capsule, kCapsuleName));
}

TrainingEnvironmentBatch* GetBatchEnvironment(PyObject* capsule) {
  return static_cast<TrainingEnvironmentBatch*>(
      PyCapsule_GetPointer(capsule, kBatchCapsuleName));
}

void DestroyEnvironment(PyObject* capsule) {
  delete GetEnvironment(capsule);
}

void DestroyBatchEnvironment(PyObject* capsule) {
  delete GetBatchEnvironment(capsule);
}

PyObject* PositionToPython(const float value[3]) {
  return Py_BuildValue("(fff)", value[0], value[1], value[2]);
}

PyObject* PairToPython(const float value[2]) {
  return Py_BuildValue("(ff)", value[0], value[1]);
}

PyObject* PlayerToPython(const GFootballModelPlayer& player) {
  PyObject* result = PyDict_New();
  if (!result) return nullptr;
  PyObject* position = PositionToPython(player.position);
  PyObject* velocity = PositionToPython(player.velocity);
  PyObject* facing = PositionToPython(player.facing);
  PyObject* formation = PairToPython(player.formation_position);
  PyObject* dynamic_formation = PairToPython(player.dynamic_formation_position);
  PyObject* tired = PyFloat_FromDouble(player.tired_factor);
  PyObject* role = PyLong_FromLong(player.role);
  PyObject* dynamic_role = PyLong_FromLong(player.dynamic_role);
  PyObject* function_type = PyLong_FromLong(player.function_type);
  PyObject* action_frame = PyLong_FromLong(player.action_frame);
  PyObject* touch_frame = PyLong_FromLong(player.touch_frame);
  PyObject* possession_duration =
      PyLong_FromLong(player.possession_duration_ms);
  PyObject* time_to_ball = PyLong_FromLong(player.time_to_ball_ms);
  PyObject* card = PyBool_FromLong(player.has_card);
  PyObject* active = PyBool_FromLong(player.is_active);
  PyObject* touch_pending = PyBool_FromLong(player.touch_pending);
  const bool success =
      position && velocity && facing && formation && dynamic_formation &&
      tired && role && dynamic_role && function_type && action_frame &&
      touch_frame && possession_duration && time_to_ball && card && active &&
      touch_pending &&
      PyDict_SetItemString(result, "position", position) == 0 &&
      PyDict_SetItemString(result, "velocity", velocity) == 0 &&
      PyDict_SetItemString(result, "facing", facing) == 0 &&
      PyDict_SetItemString(result, "formation_position", formation) == 0 &&
      PyDict_SetItemString(result, "dynamic_formation_position",
                           dynamic_formation) == 0 &&
      PyDict_SetItemString(result, "tired_factor", tired) == 0 &&
      PyDict_SetItemString(result, "role", role) == 0 &&
      PyDict_SetItemString(result, "dynamic_role", dynamic_role) == 0 &&
      PyDict_SetItemString(result, "function_type", function_type) == 0 &&
      PyDict_SetItemString(result, "action_frame", action_frame) == 0 &&
      PyDict_SetItemString(result, "touch_frame", touch_frame) == 0 &&
      PyDict_SetItemString(result, "possession_duration_ms",
                           possession_duration) == 0 &&
      PyDict_SetItemString(result, "time_to_ball_ms", time_to_ball) == 0 &&
      PyDict_SetItemString(result, "has_card", card) == 0 &&
      PyDict_SetItemString(result, "is_active", active) == 0 &&
      PyDict_SetItemString(result, "touch_pending", touch_pending) == 0;
  if (!success) {
    Py_XDECREF(position);
    Py_XDECREF(velocity);
    Py_XDECREF(facing);
    Py_XDECREF(formation);
    Py_XDECREF(dynamic_formation);
    Py_XDECREF(tired);
    Py_XDECREF(role);
    Py_XDECREF(dynamic_role);
    Py_XDECREF(function_type);
    Py_XDECREF(action_frame);
    Py_XDECREF(touch_frame);
    Py_XDECREF(possession_duration);
    Py_XDECREF(time_to_ball);
    Py_XDECREF(card);
    Py_XDECREF(active);
    Py_XDECREF(touch_pending);
    Py_DECREF(result);
    return nullptr;
  }
  Py_DECREF(position);
  Py_DECREF(velocity);
  Py_DECREF(facing);
  Py_DECREF(formation);
  Py_DECREF(dynamic_formation);
  Py_DECREF(tired);
  Py_DECREF(role);
  Py_DECREF(dynamic_role);
  Py_DECREF(function_type);
  Py_DECREF(action_frame);
  Py_DECREF(touch_frame);
  Py_DECREF(possession_duration);
  Py_DECREF(time_to_ball);
  Py_DECREF(card);
  Py_DECREF(active);
  Py_DECREF(touch_pending);
  return result;
}

PyObject* TeamToPython(
    const GFootballModelPlayer team[kGFootballPlayersPerTeam]) {
  PyObject* result = PyList_New(kGFootballPlayersPerTeam);
  if (!result) return nullptr;
  for (int player = 0; player < kGFootballPlayersPerTeam; ++player) {
    PyObject* value = PlayerToPython(team[player]);
    if (!value) {
      Py_DECREF(result);
      return nullptr;
    }
    PyList_SET_ITEM(result, player, value);
  }
  return result;
}

bool SetItem(PyObject* dictionary, const char* name, PyObject* value) {
  if (!value) return false;
  const int result = PyDict_SetItemString(dictionary, name, value);
  Py_DECREF(value);
  return result == 0;
}

PyObject* TeamStateToPython(const GFootballModelTeamState& team) {
  PyObject* result = PyDict_New();
  if (!result) return nullptr;
  bool success = true;
  success = SetItem(result, "possession_amount",
                    PyFloat_FromDouble(team.possession_amount)) && success;
  success = SetItem(result, "fading_possession_amount",
                    PyFloat_FromDouble(team.fading_possession_amount)) &&
            success;
  success = SetItem(result, "offside_trap_x",
                    PyFloat_FromDouble(team.offside_trap_x)) && success;
  success = SetItem(result, "designated_possession_player",
                    PyLong_FromLong(team.designated_possession_player)) &&
            success;
  success = SetItem(result, "time_to_ball_ms",
                    PyLong_FromLong(team.time_to_ball_ms)) && success;
  if (!success) {
    Py_DECREF(result);
    return nullptr;
  }
  return result;
}

PyObject* ObservationToPython(const TrainingObservation& observation) {
  PyObject* result = PyDict_New();
  if (!result) return nullptr;
  PyObject* teams = PyTuple_New(2);
  PyObject* team_states = PyTuple_New(2);
  PyObject* sticky = PyTuple_New(kGFootballPlayersPerTeam);
  if (!teams || !team_states || !sticky) {
    Py_XDECREF(teams);
    Py_XDECREF(team_states);
    Py_XDECREF(sticky);
    Py_DECREF(result);
    return nullptr;
  }
  for (int side = 0; side < 2; ++side) {
    PyObject* team = TeamToPython(observation.teams[side]);
    PyObject* team_state = TeamStateToPython(observation.team_state[side]);
    if (!team || !team_state) {
      Py_DECREF(teams);
      Py_DECREF(team_states);
      Py_DECREF(sticky);
      Py_XDECREF(team);
      Py_XDECREF(team_state);
      Py_DECREF(result);
      return nullptr;
    }
    PyTuple_SET_ITEM(teams, side, team);
    PyTuple_SET_ITEM(team_states, side, team_state);
  }
  for (int player = 0; player < kGFootballPlayersPerTeam; ++player) {
    PyObject* player_sticky = PyTuple_New(kGFootballStickyActionCount);
    if (!player_sticky) {
      Py_DECREF(teams);
      Py_DECREF(team_states);
      Py_DECREF(sticky);
      Py_DECREF(result);
      return nullptr;
    }
    for (int action = 0; action < kGFootballStickyActionCount; ++action) {
      PyTuple_SetItem(player_sticky, action,
                      PyBool_FromLong(observation.sticky_actions[player][action]));
    }
    PyTuple_SetItem(sticky, player, player_sticky);
  }
  bool success = true;
  auto add = [&](const char* name, PyObject* value) {
    if (!value) {
      success = false;
      return;
    }
    success = PyDict_SetItemString(result, name, value) == 0 && success;
    Py_DECREF(value);
  };
  add("ball_position", PositionToPython(observation.ball_position));
  add("ball_velocity", PositionToPython(observation.ball_velocity));
  add("ball_rotation", PositionToPython(observation.ball_rotation));
  add("teams", teams);
  teams = nullptr;
  add("team_state", team_states);
  team_states = nullptr;
  add("goals", Py_BuildValue("(ii)", observation.goals[0], observation.goals[1]));
  add("game_mode", PyLong_FromLong(observation.game_mode));
  add("set_piece_team", PyLong_FromLong(observation.set_piece_team));
  add("set_piece_taker", PyLong_FromLong(observation.set_piece_taker));
  add("ball_owned_team", PyLong_FromLong(observation.ball_owned_team));
  add("ball_owned_player", PyLong_FromLong(observation.ball_owned_player));
  add("last_touch_team", PyLong_FromLong(observation.last_touch_team));
  add("last_touch_player", PyLong_FromLong(observation.last_touch_player));
  add("match_time_ms", PyLong_FromLong(observation.match_time_ms));
  add("step", PyLong_FromLong(observation.step));
  add("is_in_play", PyBool_FromLong(observation.is_in_play));
  add("sticky_actions", sticky);
  sticky = nullptr;
  Py_XDECREF(teams);
  Py_XDECREF(team_states);
  Py_XDECREF(sticky);
  if (!success) {
    Py_DECREF(result);
    return nullptr;
  }
  return result;
}

template <typename Function>
PyObject* TranslateExceptions(Function function) {
  try {
    return function();
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
}

template <typename Function>
auto RunWithoutGil(Function function) -> decltype(function()) {
  PyThreadState* state = PyEval_SaveThread();
  try {
    auto result = function();
    PyEval_RestoreThread(state);
    return result;
  } catch (...) {
    PyEval_RestoreThread(state);
    throw;
  }
}

PyObject* Create(PyObject*, PyObject* arguments, PyObject* keywords) {
  const char* data_directory = nullptr;
  const char* font_file = nullptr;
  int maximum_steps = 3000;
  static const char* names[] = {
      "data_directory", "font_file", "maximum_steps", nullptr};
  if (!PyArg_ParseTupleAndKeywords(arguments, keywords, "ss|i",
                                   const_cast<char**>(names), &data_directory,
                                   &font_file, &maximum_steps)) {
    return nullptr;
  }
  return TranslateExceptions([&]() -> PyObject* {
    auto environment = std::make_unique<TrainingEnvironment>(
        std::filesystem::u8path(data_directory),
        std::filesystem::u8path(font_file), maximum_steps);
    PyObject* capsule = PyCapsule_New(environment.get(), kCapsuleName,
                                      DestroyEnvironment);
    if (capsule) environment.release();
    return capsule;
  });
}

PyObject* CreateBatch(PyObject*, PyObject* arguments, PyObject* keywords) {
  const char* data_directory = nullptr;
  const char* font_file = nullptr;
  int maximum_steps = 3000;
  int count = 1;
  static const char* names[] = {
      "data_directory", "font_file", "maximum_steps", "count", nullptr};
  if (!PyArg_ParseTupleAndKeywords(arguments, keywords, "ss|ii",
                                   const_cast<char**>(names), &data_directory,
                                   &font_file, &maximum_steps, &count)) {
    return nullptr;
  }
  return TranslateExceptions([&]() -> PyObject* {
    auto environment = std::make_unique<TrainingEnvironmentBatch>(
        std::filesystem::u8path(data_directory),
        std::filesystem::u8path(font_file), maximum_steps, count);
    PyObject* capsule = PyCapsule_New(environment.get(), kBatchCapsuleName,
                                      DestroyBatchEnvironment);
    if (capsule) environment.release();
    return capsule;
  });
}

PyObject* Reset(PyObject*, PyObject* arguments) {
  PyObject* capsule = nullptr;
  unsigned long seed = 0;
  int left_team = 1;
  if (!PyArg_ParseTuple(arguments, "Ok|p", &capsule, &seed, &left_team)) {
    return nullptr;
  }
  TrainingEnvironment* environment = GetEnvironment(capsule);
  if (!environment) return nullptr;
  return TranslateExceptions([&]() {
    return ObservationToPython(
        environment->Reset(static_cast<std::uint32_t>(seed), left_team != 0));
  });
}

PyObject* ResetBuiltin(PyObject*, PyObject* arguments) {
  PyObject* capsule = nullptr;
  unsigned long seed = 0;
  int left_team = 1;
  if (!PyArg_ParseTuple(arguments, "Ok|p", &capsule, &seed, &left_team)) {
    return nullptr;
  }
  TrainingEnvironment* environment = GetEnvironment(capsule);
  if (!environment) return nullptr;
  return TranslateExceptions([&]() {
    return ObservationToPython(environment->ResetBuiltin(
        static_cast<std::uint32_t>(seed), left_team != 0));
  });
}

PyObject* Step(PyObject*, PyObject* arguments) {
  PyObject* capsule = nullptr;
  PyObject* decision_object = nullptr;
  if (!PyArg_ParseTuple(arguments, "OO", &capsule, &decision_object)) {
    return nullptr;
  }
  TrainingEnvironment* environment = GetEnvironment(capsule);
  if (!environment) return nullptr;
  PyObject* sequence = PySequence_Fast(
      decision_object, "decision must be a sequence of eleven actions");
  if (!sequence) return nullptr;
  if (PySequence_Fast_GET_SIZE(sequence) != kGFootballPlayersPerTeam) {
    Py_DECREF(sequence);
    PyErr_SetString(PyExc_ValueError, "decision must contain eleven actions");
    return nullptr;
  }
  std::array<std::int32_t, kGFootballPlayersPerTeam> decision{};
  for (int index = 0; index < kGFootballPlayersPerTeam; ++index) {
    const long action = PyLong_AsLong(PySequence_Fast_GET_ITEM(sequence, index));
    if (action == -1 && PyErr_Occurred()) {
      Py_DECREF(sequence);
      return nullptr;
    }
    decision[index] = static_cast<std::int32_t>(action);
  }
  Py_DECREF(sequence);
  return TranslateExceptions([&]() {
    const TrainingObservation observation =
        environment->Step(decision);
    PyObject* result = PyTuple_New(2);
    PyObject* state = ObservationToPython(observation);
    PyObject* terminated = PyBool_FromLong(
        observation.step >= environment->maximum_steps());
    if (!result || !state || !terminated) {
      Py_XDECREF(result);
      Py_XDECREF(state);
      Py_XDECREF(terminated);
      return static_cast<PyObject*>(nullptr);
    }
    PyTuple_SET_ITEM(result, 0, state);
    PyTuple_SET_ITEM(result, 1, terminated);
    return result;
  });
}

PyObject* ResetBatch(PyObject*, PyObject* arguments) {
  PyObject* capsule = nullptr;
  PyObject* seed_object = nullptr;
  PyObject* side_object = nullptr;
  if (!PyArg_ParseTuple(arguments, "OOO", &capsule, &seed_object,
                        &side_object)) {
    return nullptr;
  }
  TrainingEnvironmentBatch* environment = GetBatchEnvironment(capsule);
  if (!environment) return nullptr;
  PyObject* seeds = PySequence_Fast(
      seed_object, "seeds must be a sequence with one value per environment");
  PyObject* sides = PySequence_Fast(
      side_object, "sides must be a sequence with one value per environment");
  if (!seeds || !sides) {
    Py_XDECREF(seeds);
    Py_XDECREF(sides);
    return nullptr;
  }
  const int count = environment->size();
  if (PySequence_Fast_GET_SIZE(seeds) != count ||
      PySequence_Fast_GET_SIZE(sides) != count) {
    Py_DECREF(seeds);
    Py_DECREF(sides);
    PyErr_SetString(PyExc_ValueError,
                    "batch reset requires one seed and side per environment");
    return nullptr;
  }
  std::vector<std::uint32_t> seed_values(count);
  std::vector<bool> side_values(count);
  for (int index = 0; index < count; ++index) {
    const unsigned long long seed = PyLong_AsUnsignedLongLong(
        PySequence_Fast_GET_ITEM(seeds, index));
    const int side = PyObject_IsTrue(PySequence_Fast_GET_ITEM(sides, index));
    if (PyErr_Occurred() || seed > UINT32_MAX || side < 0) {
      Py_DECREF(seeds);
      Py_DECREF(sides);
      if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_ValueError, "seed must fit in 32 bits");
      }
      return nullptr;
    }
    seed_values[index] = static_cast<std::uint32_t>(seed);
    side_values[index] = side != 0;
  }
  Py_DECREF(seeds);
  Py_DECREF(sides);
  return TranslateExceptions([&]() -> PyObject* {
    const std::vector<TrainingObservation> observations = RunWithoutGil(
        [&]() { return environment->Reset(seed_values, side_values); });
    PyObject* result = PyList_New(count);
    if (!result) return nullptr;
    for (int index = 0; index < count; ++index) {
      PyObject* observation = ObservationToPython(observations[index]);
      if (!observation) {
        Py_DECREF(result);
        return nullptr;
      }
      PyList_SET_ITEM(result, index, observation);
    }
    return result;
  });
}

PyObject* ResetBuiltinBatch(PyObject*, PyObject* arguments) {
  PyObject* capsule = nullptr;
  PyObject* seed_object = nullptr;
  PyObject* side_object = nullptr;
  if (!PyArg_ParseTuple(arguments, "OOO", &capsule, &seed_object,
                        &side_object)) {
    return nullptr;
  }
  TrainingEnvironmentBatch* environment = GetBatchEnvironment(capsule);
  if (!environment) return nullptr;
  PyObject* seeds = PySequence_Fast(
      seed_object, "seeds must be a sequence with one value per environment");
  PyObject* sides = PySequence_Fast(
      side_object, "sides must be a sequence with one value per environment");
  if (!seeds || !sides) {
    Py_XDECREF(seeds);
    Py_XDECREF(sides);
    return nullptr;
  }
  const int count = environment->size();
  if (PySequence_Fast_GET_SIZE(seeds) != count ||
      PySequence_Fast_GET_SIZE(sides) != count) {
    Py_DECREF(seeds);
    Py_DECREF(sides);
    PyErr_SetString(PyExc_ValueError,
                    "batch reset requires one seed and side per environment");
    return nullptr;
  }
  std::vector<std::uint32_t> seed_values(count);
  std::vector<bool> side_values(count);
  for (int index = 0; index < count; ++index) {
    const unsigned long long seed = PyLong_AsUnsignedLongLong(
        PySequence_Fast_GET_ITEM(seeds, index));
    const int side = PyObject_IsTrue(PySequence_Fast_GET_ITEM(sides, index));
    if (PyErr_Occurred() || seed > UINT32_MAX || side < 0) {
      Py_DECREF(seeds);
      Py_DECREF(sides);
      if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_ValueError, "seed must fit in 32 bits");
      }
      return nullptr;
    }
    seed_values[index] = static_cast<std::uint32_t>(seed);
    side_values[index] = side != 0;
  }
  Py_DECREF(seeds);
  Py_DECREF(sides);
  return TranslateExceptions([&]() -> PyObject* {
    const std::vector<TrainingObservation> observations = RunWithoutGil(
        [&]() { return environment->ResetBuiltin(seed_values, side_values); });
    PyObject* result = PyList_New(count);
    if (!result) return nullptr;
    for (int index = 0; index < count; ++index) {
      PyObject* observation = ObservationToPython(observations[index]);
      if (!observation) {
        Py_DECREF(result);
        return nullptr;
      }
      PyList_SET_ITEM(result, index, observation);
    }
    return result;
  });
}

PyObject* ResetBatchOne(PyObject*, PyObject* arguments) {
  PyObject* capsule = nullptr;
  int index = 0;
  unsigned long long seed = 0;
  int left_team = 1;
  if (!PyArg_ParseTuple(arguments, "OiK|p", &capsule, &index, &seed,
                        &left_team)) {
    return nullptr;
  }
  if (seed > UINT32_MAX) {
    PyErr_SetString(PyExc_ValueError, "seed must fit in 32 bits");
    return nullptr;
  }
  TrainingEnvironmentBatch* environment = GetBatchEnvironment(capsule);
  if (!environment) return nullptr;
  return TranslateExceptions([&]() {
    const TrainingObservation observation = RunWithoutGil([&]() {
      return environment->ResetOne(index, static_cast<std::uint32_t>(seed),
                                   left_team != 0);
    });
    return ObservationToPython(observation);
  });
}

PyObject* StepBatch(PyObject*, PyObject* arguments) {
  PyObject* capsule = nullptr;
  PyObject* decisions_object = nullptr;
  if (!PyArg_ParseTuple(arguments, "OO", &capsule, &decisions_object)) {
    return nullptr;
  }
  TrainingEnvironmentBatch* environment = GetBatchEnvironment(capsule);
  if (!environment) return nullptr;
  PyObject* decisions = PySequence_Fast(
      decisions_object, "decisions must contain one decision per environment");
  if (!decisions) return nullptr;
  const int count = environment->size();
  if (PySequence_Fast_GET_SIZE(decisions) != count) {
    Py_DECREF(decisions);
    PyErr_SetString(PyExc_ValueError,
                    "batch step requires one decision per environment");
    return nullptr;
  }
  std::vector<std::array<std::int32_t, kGFootballPlayersPerTeam>> values(count);
  for (int environment_index = 0; environment_index < count;
       ++environment_index) {
    PyObject* decision = PySequence_Fast(
        PySequence_Fast_GET_ITEM(decisions, environment_index),
        "each decision must contain eleven actions");
    if (!decision) {
      Py_DECREF(decisions);
      return nullptr;
    }
    if (PySequence_Fast_GET_SIZE(decision) != kGFootballPlayersPerTeam) {
      Py_DECREF(decision);
      Py_DECREF(decisions);
      PyErr_SetString(PyExc_ValueError,
                      "each decision must contain eleven actions");
      return nullptr;
    }
    for (int player = 0; player < kGFootballPlayersPerTeam; ++player) {
      const long action =
          PyLong_AsLong(PySequence_Fast_GET_ITEM(decision, player));
      if (action == -1 && PyErr_Occurred()) {
        Py_DECREF(decision);
        Py_DECREF(decisions);
        return nullptr;
      }
      values[environment_index][player] = static_cast<std::int32_t>(action);
    }
    Py_DECREF(decision);
  }
  Py_DECREF(decisions);
  return TranslateExceptions([&]() -> PyObject* {
    const std::vector<TrainingObservation> observations =
        RunWithoutGil([&]() { return environment->Step(values); });
    PyObject* result = PyList_New(count);
    if (!result) return nullptr;
    for (int index = 0; index < count; ++index) {
      PyObject* pair = PyTuple_New(2);
      PyObject* observation = ObservationToPython(observations[index]);
      PyObject* terminated = PyBool_FromLong(
          observations[index].step >= environment->maximum_steps());
      if (!pair || !observation || !terminated) {
        Py_XDECREF(pair);
        Py_XDECREF(observation);
        Py_XDECREF(terminated);
        Py_DECREF(result);
        return nullptr;
      }
      PyTuple_SET_ITEM(pair, 0, observation);
      PyTuple_SET_ITEM(pair, 1, terminated);
      PyList_SET_ITEM(result, index, pair);
    }
    return result;
  });
}

PyMethodDef methods[] = {
    {"create", reinterpret_cast<PyCFunction>(Create),
     METH_VARARGS | METH_KEYWORDS, "Create one native training environment."},
    {"reset", Reset, METH_VARARGS, "Reset a match with the requested seed."},
    {"reset_builtin", ResetBuiltin, METH_VARARGS,
     "Reset a pure built-in-AI match for demonstration collection."},
    {"step", Step, METH_VARARGS,
     "Submit one player/action decision and advance the match."},
    {"create_batch", reinterpret_cast<PyCFunction>(CreateBatch),
     METH_VARARGS | METH_KEYWORDS, "Create native training environments."},
    {"reset_batch", ResetBatch, METH_VARARGS,
     "Reset every native training environment."},
    {"reset_builtin_batch", ResetBuiltinBatch, METH_VARARGS,
     "Reset native environments without external controllers."},
    {"reset_batch_one", ResetBatchOne, METH_VARARGS,
     "Reset one native training environment."},
    {"step_batch", StepBatch, METH_VARARGS,
     "Advance native training environments concurrently."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_gfootball_env",
    "Native gfootball training environment.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__gfootball_env() {
  PyObject* result = PyModule_Create(&module);
  if (!result) return nullptr;
  if (PyModule_AddIntConstant(result, "ENGINE_ACTION_COUNT",
                              game_delegate) < 0) {
    Py_DECREF(result);
    return nullptr;
  }
  return result;
}
