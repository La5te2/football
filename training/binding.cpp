// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

// CPython bindings for the native reinforcement-learning environment.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>

#include "environment.hpp"

namespace {

constexpr char kCapsuleName[] = "gfootball.TrainingEnvironment";

TrainingEnvironment* GetEnvironment(PyObject* capsule) {
  return static_cast<TrainingEnvironment*>(
      PyCapsule_GetPointer(capsule, kCapsuleName));
}

void DestroyEnvironment(PyObject* capsule) {
  delete GetEnvironment(capsule);
}

PyObject* PositionToPython(const std::array<float, 3>& value) {
  return Py_BuildValue("(fff)", value[0], value[1], value[2]);
}

PyObject* PlayerToPython(const TrainingPlayer& player) {
  PyObject* result = PyDict_New();
  if (!result) return nullptr;
  PyObject* position = PositionToPython(player.position);
  PyObject* direction = PositionToPython(player.direction);
  PyObject* tired = PyFloat_FromDouble(player.tired_factor);
  PyObject* role = PyLong_FromLong(player.role);
  PyObject* card = PyBool_FromLong(player.has_card);
  PyObject* active = PyBool_FromLong(player.is_active);
  if (!position || !direction || !tired || !role || !card || !active ||
      PyDict_SetItemString(result, "position", position) < 0 ||
      PyDict_SetItemString(result, "direction", direction) < 0 ||
      PyDict_SetItemString(result, "tired_factor", tired) < 0 ||
      PyDict_SetItemString(result, "role", role) < 0 ||
      PyDict_SetItemString(result, "has_card", card) < 0 ||
      PyDict_SetItemString(result, "is_active", active) < 0) {
    Py_XDECREF(position);
    Py_XDECREF(direction);
    Py_XDECREF(tired);
    Py_XDECREF(role);
    Py_XDECREF(card);
    Py_XDECREF(active);
    Py_DECREF(result);
    return nullptr;
  }
  Py_DECREF(position);
  Py_DECREF(direction);
  Py_DECREF(tired);
  Py_DECREF(role);
  Py_DECREF(card);
  Py_DECREF(active);
  return result;
}

PyObject* TeamToPython(
    const std::array<TrainingPlayer, kTrainingPlayersPerTeam>& team) {
  PyObject* result = PyList_New(kTrainingPlayersPerTeam);
  if (!result) return nullptr;
  for (int player = 0; player < kTrainingPlayersPerTeam; ++player) {
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

PyObject* ObservationToPython(const TrainingObservation& observation) {
  PyObject* result = PyDict_New();
  if (!result) return nullptr;
  PyObject* teams = PyTuple_New(2);
  PyObject* sticky = PyTuple_New(kTrainingStickyActionCount);
  if (!teams || !sticky) {
    Py_XDECREF(teams);
    Py_XDECREF(sticky);
    Py_DECREF(result);
    return nullptr;
  }
  for (int side = 0; side < 2; ++side) {
    PyObject* team = TeamToPython(observation.teams[side]);
    if (!team) {
      Py_DECREF(teams);
      Py_DECREF(sticky);
      Py_DECREF(result);
      return nullptr;
    }
    PyTuple_SET_ITEM(teams, side, team);
  }
  for (int index = 0; index < kTrainingStickyActionCount; ++index) {
    PyObject* value = PyBool_FromLong(observation.sticky_actions[index]);
    if (!value) {
      Py_DECREF(teams);
      Py_DECREF(sticky);
      Py_DECREF(result);
      return nullptr;
    }
    PyTuple_SET_ITEM(sticky, index, value);
  }
  const bool success =
      SetItem(result, "ball_position",
              PositionToPython(observation.ball_position)) &&
      SetItem(result, "ball_direction",
              PositionToPython(observation.ball_direction)) &&
      SetItem(result, "ball_rotation",
              PositionToPython(observation.ball_rotation)) &&
      SetItem(result, "teams", teams) &&
      SetItem(result, "goals",
              Py_BuildValue("(ii)", observation.goals[0],
                            observation.goals[1])) &&
      SetItem(result, "game_mode",
              PyLong_FromLong(observation.game_mode)) &&
      SetItem(result, "ball_owned_team",
              PyLong_FromLong(observation.ball_owned_team)) &&
      SetItem(result, "ball_owned_player",
              PyLong_FromLong(observation.ball_owned_player)) &&
      SetItem(result, "step", PyLong_FromLong(observation.step)) &&
      SetItem(result, "sticky_actions", sticky);
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

PyObject* Reset(PyObject*, PyObject* arguments) {
  PyObject* capsule = nullptr;
  unsigned long seed = 0;
  if (!PyArg_ParseTuple(arguments, "Ok", &capsule, &seed)) return nullptr;
  TrainingEnvironment* environment = GetEnvironment(capsule);
  if (!environment) return nullptr;
  return TranslateExceptions([&]() {
    return ObservationToPython(
        environment->Reset(static_cast<std::uint32_t>(seed)));
  });
}

PyObject* Step(PyObject*, PyObject* arguments) {
  PyObject* capsule = nullptr;
  int controlled_player = 0;
  int action = 0;
  if (!PyArg_ParseTuple(arguments, "Oii", &capsule, &controlled_player,
                        &action)) {
    return nullptr;
  }
  TrainingEnvironment* environment = GetEnvironment(capsule);
  if (!environment) return nullptr;
  return TranslateExceptions([&]() {
    const TrainingObservation observation =
        environment->Step(controlled_player, action);
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

PyMethodDef methods[] = {
    {"create", reinterpret_cast<PyCFunction>(Create),
     METH_VARARGS | METH_KEYWORDS, "Create one native training environment."},
    {"reset", Reset, METH_VARARGS, "Reset a match with the requested seed."},
    {"step", Step, METH_VARARGS,
     "Submit one player/action decision and advance the match."},
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
  if (PyModule_AddIntConstant(result, "PLAYER_COUNT",
                              kTrainingPlayersPerTeam) < 0 ||
      PyModule_AddIntConstant(result, "ACTION_COUNT",
                              kTrainingActionCount) < 0) {
    Py_DECREF(result);
    return nullptr;
  }
  return result;
}
