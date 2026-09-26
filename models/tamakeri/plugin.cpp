// Copyright 2026 gfootball contributors
// Licensed under the Apache License, Version 2.0.

// Dynamic-library entry points for the TamakEri model plugin.

#include "adapter.hpp"

#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>

#ifdef _WIN32
#define GFOOTBALL_MODEL_EXPORT extern "C" __declspec(dllexport)
#else
#define GFOOTBALL_MODEL_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

void SetError(char* destination, std::size_t capacity, const char* message) {
  if (!destination || capacity == 0) return;
#ifdef _WIN32
  strncpy_s(destination, capacity, message, _TRUNCATE);
#else
  std::strncpy(destination, message, capacity - 1);
  destination[capacity - 1] = '\0';
#endif
}

}  // namespace

// Resolves weights relative to the DLL and creates an opaque adapter instance.
GFOOTBALL_MODEL_EXPORT GFootballModelHandle gfootball_model_create(
    const char* model_directory, std::int32_t side,
    std::int32_t game_duration, char* error, std::size_t error_capacity) {
  try {
    const auto path =
        std::filesystem::path(model_directory) / "tamakeri.pt";
    return new TamakEriAdapter(path.string(), side, game_duration);
  } catch (const std::exception& exception) {
    SetError(error, error_capacity, exception.what());
    return nullptr;
  }
}

// Destroys the adapter in the DLL that allocated it.
GFOOTBALL_MODEL_EXPORT void gfootball_model_destroy(
    GFootballModelHandle model) {
  delete static_cast<TamakEriAdapter*>(model);
}

// Clears all per-match adapter state.
GFOOTBALL_MODEL_EXPORT void gfootball_model_reset(
    GFootballModelHandle model) {
  static_cast<TamakEriAdapter*>(model)->Reset();
}

// Forwards one observation and reports exceptions through the C error buffer.
GFOOTBALL_MODEL_EXPORT std::int32_t gfootball_model_decide(
    GFootballModelHandle model,
    const GFootballModelObservation* observation,
    GFootballModelDecision* decision, char* error,
    std::size_t error_capacity) {
  try {
    if (!model || !observation || !decision) {
      throw std::invalid_argument("Invalid model inference argument");
    }
    *decision = static_cast<TamakEriAdapter*>(model)->Decide(*observation);
    return 1;
  } catch (const std::exception& exception) {
    SetError(error, error_capacity, exception.what());
    return 0;
  }
}
