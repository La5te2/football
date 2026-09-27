// Copyright 2019 Google LLC & Bastiaan Konings
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// written by bastiaan konings schuiling 2008 - 2015
// this work is public domain. the code is undocumented, scruffy, untested, and
// should generally not be used for anything important. i do not offer support,
// so don't ask. to be used for inspiration :)

#include "team.hpp"

#include "../gamedefines.hpp"
#include "../main.hpp"
#include "../utils.hpp"
#include "AIsupport/AIfunctions.hpp"
#include "match.hpp"

Team::Team(int id, Match *match, TeamData *teamData)
    : id(id), match(match), teamData(teamData) {
  DO_VALIDATION;
  assert(id == 0 || id == 1);
  // assert(teamData->GetPlayerNum() >= playerNum); // does team have enough
  // players?

  teamNode =
      boost::intrusive_ptr<Node>(new Node("team node #" + int_to_str(id)));
  teamNode->SetLocalMode(e_LocalMode_Absolute);
  match->GetDynamicNode()->AddNode(teamNode);

  teamController = new TeamAIController(this);

  timeNeededToGetToBall_ms = 100;
  hasPossession = false;

  teamPossessionAmount = 1.0;
  fadingTeamPossessionAmount = 1.0;
}

Team::~Team() { DO_VALIDATION; }

void Team::Mirror() {
  side *= -1;
  mirrored = !mirrored;
  for (auto &p : players) {
    p->Mirror();
  }
}

void Team::Exit() {
  DO_VALIDATION;
  Hide2D();

  externalControllers.clear();
  for (unsigned int i = 0; i < players.size(); i++) {
    DO_VALIDATION;
    delete players[i];
  }

  delete teamController;

  playerNode->Exit();
  playerNode.reset();

  match->GetDynamicNode()->DeleteNode(teamNode);
}

void Team::InitPlayers(boost::intrusive_ptr<Node> fullbodyNode,
                       std::map<Vector3, Vector3> &colorCoords) {
  DO_VALIDATION;
  // first, load 1 instance of a player

  ObjectLoader loader;
  playerNode = loader.LoadObject("media/objects/players/player.object");
  playerNode->SetName("player");
  playerNode->SetLocalMode(e_LocalMode_Absolute);

  // load all players in the team, even the players who sit on the bench. aww.
  for (int i = 0; i < (signed int)teamData->GetPlayerNum(); i++) {
    DO_VALIDATION;
    PlayerData *playerData = teamData->GetPlayerData(i);
    Player *player = new Player(this, playerData);
    players.push_back(player);

    if (i < playerNum) {
      DO_VALIDATION;
      // activate playerCount players (the starting eleven, usually)
      std::string kitFilename;
      // printf("%i player id\n", player->GetID());
      auto formation = GetFormationEntry(player);
      if (formation.role != e_PlayerRole_GK) {
        DO_VALIDATION;
        kitFilename = GetTeamData()->GetKitUrl() + "_kit_0" +
                      int_to_str(GetMenuTask()->GetTeamKitNum(GetID())) +
                      ".png";
      } else {
        kitFilename = "media/objects/players/textures/goalie_kit.png";
      }
      kit = GetContext().surface_manager.Fetch(kitFilename);
      player->Activate(
          playerNode, fullbodyNode,
          colorCoords, kit, match->GetAnimCollection(), formation.lazy);
    }
  }

  designatedTeamPossessionPlayer = players.at(0);
}

FormationEntry Team::GetFormationEntry(void *player) {
  DO_VALIDATION;
  for (int i = 0; i < (signed int)players.size(); i++) {
    DO_VALIDATION;
    if (players[i] == player) {
      DO_VALIDATION;
      return teamData->GetFormationEntry(i);
    }
  }

  assert(1 == 2);
  FormationEntry fail;
  return fail;
}

void Team::SetFormationEntry(Player *player, FormationEntry entry) {
  DO_VALIDATION;
  for (int i = 0; i < (signed int)players.size(); i++) {
    DO_VALIDATION;
    if (players[i] == player) {
      DO_VALIDATION;
      teamData->SetFormationEntry(i, entry);
    }
  }
}

void Team::GetActivePlayers(std::vector<Player *> &activePlayers) {
  DO_VALIDATION;
  for (auto player : players) {
    DO_VALIDATION;
    if (player->IsActive()) activePlayers.push_back(player);
  }
}

int Team::GetActivePlayersCount() const {
  int count = 0;
  for (auto player : players) {
    DO_VALIDATION;
    if (player->IsActive()) count++;
  }
  return count;
}

void Team::AddExternalControllers(const std::vector<ExternalInput*>& inputs) {
  DO_VALIDATION;
  for (auto input : inputs) {
    input->SetDisabled(true);
    externalControllers.push_back(
        std::make_unique<ExternalController>(this, input));
  }
  UpdateDesignatedTeamPossessionPlayer();
}

bool Team::SetExternalControllerPlayer(int controller_index, int player_index) {
  DO_VALIDATION;
  if (controller_index < 0 ||
      controller_index >= static_cast<int>(externalControllers.size())) {
    return false;
  }
  auto* controller = externalControllers[controller_index].get();
  if (player_index < 0) {
    if (mainSelectedPlayer == controller->GetSelectedPlayer()) {
      mainSelectedPlayer = nullptr;
    }
    controller->SetSelectedPlayer(nullptr);
    return true;
  }
  if (player_index >= static_cast<int>(players.size())) {
    return false;
  }
  Player* player = players[player_index];
  if (!player->IsActive() || !player->GetFormationEntry().controllable ||
      (player->GetExternalExecutor() &&
       player->GetExternalExecutor() != controller->GetExternalExecutor())) {
    return false;
  }
  controller->SetSelectedPlayer(player);
  mainSelectedPlayer = player;
  return controller->GetSelectedPlayer() == player;
}

void Team::UpdateDesignatedTeamPossessionPlayer() {
  DO_VALIDATION;
  designatedTeamPossessionPlayer =
      AI_GetClosestPlayer(this, match->GetBall()->Predict(0).Get2D(), false);
}

e_PlayerColor Team::GetPlayerColor(PlayerBase *player) {
  DO_VALIDATION;
  if (player->GetExternalExecutor()) {
    return player->GetExternalExecutor()->GetInput()->GetPlayerColor();
  }
  return e_PlayerColor_Default;
}

bool Team::HasPossession() const { return hasPossession; }

bool Team::HasUniquePossession() const {
  return HasPossession() && !match->GetTeam(1 - id)->HasPossession();
}

int Team::GetTimeNeededToGetToBall_ms() const {
  return timeNeededToGetToBall_ms;
}

Player *Team::GetBestPossessionPlayer() {
  DO_VALIDATION;
  int bestTime_ms = 10000000;
  Player *bestPlayer = 0;
  for (auto p : players) {
    DO_VALIDATION;
    if (p->IsActive()) {
      DO_VALIDATION;
      int time_ms = p->GetTimeNeededToGetToBall_ms();
      if (time_ms < bestTime_ms) {
        DO_VALIDATION;
        bestTime_ms = time_ms;
        bestPlayer = p;
      }
    }
  }

  assert(bestPlayer);

  return bestPlayer;
}

float Team::GetTeamPossessionAmount() const { return teamPossessionAmount; }

float Team::GetFadingTeamPossessionAmount() const {
  return fadingTeamPossessionAmount;
}

void Team::SetFadingTeamPossessionAmount(float value) {
  DO_VALIDATION;
  fadingTeamPossessionAmount = clamp(value, 0.5, 1.5);
}

void Team::SetLastTouchPlayer(Player *player, e_TouchType touchType) {
  DO_VALIDATION;
  lastTouchPlayer = player;
  player->SetLastTouchTime_ms(match->GetActualTime_ms());
  player->SetLastTouchType(touchType);
  match->SetLastTouchTeamID(GetID(), touchType);
}

void Team::ResetSituation(const Vector3 &focusPos) {
  DO_VALIDATION;
  timeNeededToGetToBall_ms = 100;
  hasPossession = false;

  teamPossessionAmount = 1.0f;
  fadingTeamPossessionAmount = 1.0f;
  lastTouchPlayer = 0;

  designatedTeamPossessionPlayer = players.at(0);

  for (unsigned int i = 0; i < players.size(); i++) {
    DO_VALIDATION;
    if (players[i]->IsActive()) {
      DO_VALIDATION;
      players[i]->ResetSituation(focusPos);
    }
  }

  GetController()->Reset();
}

void Team::SetDesignatedTeamPossessionPlayer(Player *player) {
  DO_VALIDATION;
  designatedTeamPossessionPlayer = player;
}

void Team::DeselectPlayer(Player *player) {
  DO_VALIDATION;
  for (unsigned int i = 0; i < externalControllers.size(); i++) {
    DO_VALIDATION;
    Player* selectedPlayer = externalControllers[i]->GetSelectedPlayer();
    if (selectedPlayer == player) {
      DO_VALIDATION;
      if (mainSelectedPlayer == player) {
        mainSelectedPlayer = nullptr;
      }
      externalControllers[i]->SetSelectedPlayer(nullptr);
    }
  }
}

void Team::RelaxFatigue(float howMuch) {
  DO_VALIDATION;
  for (unsigned int i = 0; i < players.size(); i++) {
    DO_VALIDATION;
    if (players[i]->IsActive()) {
      DO_VALIDATION;
      players[i]->RelaxFatigue(howMuch);
    }
  }
}

void Team::Process() {
  DO_VALIDATION;
  teamPossessionAmount = (float)(match->GetTeam(abs(GetID() - 1))
      ->GetTimeNeededToGetToBall_ms() +
      1500) /
      (float)(GetTimeNeededToGetToBall_ms() + 1500);
  float tmpFadingTeamPossessionAmount =
      fadingTeamPossessionAmount * 0.995f +
      clamp(teamPossessionAmount, 0.5f, 1.5f) * 0.005f;
  fadingTeamPossessionAmount +=
      clamp(tmpFadingTeamPossessionAmount - fadingTeamPossessionAmount,
            -0.005f, 0.005f);  // maximum change per 10ms

  if (!match->IsInPlay() || match->IsInSetPiece() ||
      match->GetBallRetainer() != 0) {
    DO_VALIDATION;
    if (match->GetBallRetainer() != 0) {
      DO_VALIDATION;
      fadingTeamPossessionAmount = teamPossessionAmount =
          (match->GetBallRetainer()->GetTeam() == this) ? 1.5f : 0.5f;
    } else {
      fadingTeamPossessionAmount = teamPossessionAmount =
          (match->GetBestPossessionTeam() == this) ? 1.5f : 0.5f;
    }
  }

  if (match->IsInPlay() && !match->IsInSetPiece()) {
    DO_VALIDATION;
    teamController->Process();

    int team_offset = id == match->SecondTeam() ? 200 : 0;
    if ((match->GetActualTime_ms() + team_offset) % 400 == 0) {
      DO_VALIDATION;
      teamController->CalculateDynamicRoles();
    }

    if ((match->GetActualTime_ms() + team_offset + 100) % 400 == 0) {
      DO_VALIDATION;
      teamController->CalculateManMarking();
    }
  }

  for (unsigned int i = 0; i < players.size(); i++) {
    DO_VALIDATION;
    if (players[i]->IsActive()) {
      DO_VALIDATION;
      players[i]->Process();
    }
  }

  int designatedPlayerTime_ms =
      designatedTeamPossessionPlayer->GetTimeNeededToGetToBall_ms();
  Player *bestPlayer = GetBestPossessionPlayer();
  int oppTime_ms =
      match->GetTeam(abs(GetID() - 1))->GetTimeNeededToGetToBall_ms();
  if (designatedTeamPossessionPlayer != bestPlayer) {
    DO_VALIDATION;
    // switch only if other player is somewhat better, to overcome
    // possession-chaos
    int bestPlayerTime_ms = bestPlayer->GetTimeNeededToGetToBall_ms();
    float timeRating = (float)(bestPlayerTime_ms + 500) /
        (float)(designatedPlayerTime_ms + 500);

    if (bestPlayer->HasPossession()) timeRating *= 0.5f;
    if (designatedTeamPossessionPlayer->HasPossession()) timeRating /= 0.5f;

    if (bestPlayer->HasActiveExternalController()) timeRating *= 0.8f;
    if (designatedTeamPossessionPlayer->HasActiveExternalController())
      timeRating /= 0.8f;

    // current player can get to the ball before the closest opponent: less
    // need to switch
    // if (GetID() == 0) printf("opptime: %i, designated time: %i\n",
    // oppTime_ms, designatedPlayerTime_ms);
    if (!bestPlayer->HasActiveExternalController() &&
        designatedPlayerTime_ms < oppTime_ms - 100) {
      DO_VALIDATION;
      timeRating += 0.2f;
      timeRating *= 1.2f;
    }

    if (timeRating < 0.8f) {
      DO_VALIDATION;
      designatedTeamPossessionPlayer = bestPlayer;
    }
  }
}

void Team::PreparePutBuffers() {
  DO_VALIDATION;
  for (unsigned int i = 0; i < players.size(); i++) {
    DO_VALIDATION;
    if (players[i]->IsActive()) {
      DO_VALIDATION;
      players[i]->PreparePutBuffers();
    }
  }
}

void Team::FetchPutBuffers() {
  DO_VALIDATION;
  for (unsigned int i = 0; i < players.size(); i++) {
    DO_VALIDATION;
    if (players[i]->IsActive()) {
      DO_VALIDATION;
      players[i]->FetchPutBuffers();
    }
  }
}

void Team::Put(bool mirror) {
  DO_VALIDATION;
  for (unsigned int i = 0; i < players.size(); i++) {
    DO_VALIDATION;
    if (players[i]->IsActive()) {
      DO_VALIDATION;
      players[i]->Put(mirror);
    }
  }
}

void Team::Put2D(bool mirror) {
  DO_VALIDATION;
  for (unsigned int i = 0; i < players.size(); i++) {
    DO_VALIDATION;
    if (players[i]->IsActive()) {
      DO_VALIDATION;
      players[i]->Put2D(mirror);
    }
  }
}

void Team::Hide2D() {
  DO_VALIDATION;
  for (unsigned int i = 0; i < players.size(); i++) {
    DO_VALIDATION;
    if (players[i]->IsActive()) {
      DO_VALIDATION;
      players[i]->Hide2D();
    }
  }
}

void Team::UpdatePossessionStats() {
  DO_VALIDATION;
  for (unsigned int i = 0; i < players.size(); i++) {
    DO_VALIDATION;
    if (players[i]->IsActive()) {
      DO_VALIDATION;
      players[i]->UpdatePossessionStats();
    }
  }

  // possession?

  hasPossession = false;
  timeNeededToGetToBall_ms = 100000;
  for (int i = 0; i < (signed int)players.size(); i++) {
    DO_VALIDATION;
    if (players[i]->IsActive()) {
      DO_VALIDATION;
      if (players[i]->HasPossession()) hasPossession = true;
      if (players[i]->GetTimeNeededToGetToBall_ms() < timeNeededToGetToBall_ms)
        timeNeededToGetToBall_ms = players[i]->GetTimeNeededToGetToBall_ms();
    }
  }
}

Player *Team::GetGoalie() {
  DO_VALIDATION;
  for (unsigned int i = 0; i < players.size(); i++) {
    DO_VALIDATION;
    if (players[i]->IsActive()) {
      DO_VALIDATION;
      if (players[i]->GetFormationEntry().role == e_PlayerRole_GK)
        return players[i];
    }
  }

  return 0;
}

void Team::ProcessState(EnvState *state) {
  DO_VALIDATION;
  state->process(hasPossession);
  state->process(timeNeededToGetToBall_ms);
  state->process(designatedTeamPossessionPlayer);
  state->process(teamPossessionAmount);
  state->process(fadingTeamPossessionAmount);
  teamController->ProcessState(state);
  int size = externalControllers.size();
  state->process(size);
  externalControllers.resize(size);
  for (auto &g : externalControllers) {
    DO_VALIDATION;
    g->ProcessState(state);
  }
  state->process(lastTouchPlayer);
  state->process(mainSelectedPlayer);
}
