from captureAgents import CaptureAgent
from game import Directions, Actions
from agents.t_047.Qfunction import QFunction
from agents.t_047.positionSearchProblem import *
import time
import math
from itertools import product

MAX_CAPACITY = 10
MAX_LEARNING_ITERATIONS = 3

class ChoiceHistory:
    def __init__(self):
        self.history = []
        self.size = 5

    def add(self, choice, q_value):
        if len(self.history) >= self.size:
            self.history.pop(0)
        self.history.append((choice, q_value))

    def get_history(self):
        return self.history
    
    def reset(self):
        self.history = []

class VIAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

        self.boundary = self.getBoundary(gameState)
        self.discount_factor = 0.9
        self.q_values = QFunction()
        self.choice_history = ChoiceHistory()
        self.choice_history.add(Directions.STOP, None)

    def getBoundary(self, gameState, opponentSide=False):
        boundary = []
        height = gameState.data.layout.height
        width = gameState.data.layout.width

        for i in range(height):
            if self.red:
                j = int(width / 2) - 1 if not opponentSide else int(width / 2)
            else:
                j = int(width / 2) if not opponentSide else int(width / 2) - 1

            if not gameState.hasWall(j, i):
                boundary.append((j, i))

        return boundary
    
    def getClosestPos(self, gameState, pos_list):
        min_length = 99999
        min_pos = None
        my_local_state = gameState.getAgentState(self.index)
        my_pos = my_local_state.getPosition()

        for pos in pos_list:
            temp_length = self.getMazeDistance(my_pos, pos)
            if temp_length < min_length:
                min_length = temp_length
                min_pos = pos

        return min_pos
    
    def getObservableOpponents(self, gameState):
        agentState = gameState.getAgentState(self.index)
        agentPos = agentState.getPosition()
        opponents = self.getOpponents(gameState)
        dangerous_opponents = []
        attackable_opponents = []

        for opponent in opponents:
            opponentPos = gameState.getAgentState(opponent).getPosition()
            if not opponentPos:
                continue

            opponentState = gameState.getAgentState(opponent)
            if (opponentState.isPacman and agentState.scaredTimer > 0) or (not opponentState.isPacman and opponentState.scaredTimer == 0):
                dangerous_opponents.append(opponent)
            elif self.getMazeDistance(agentPos, opponentPos) <= 1:
                attackable_opponents.append(opponent)

        return (dangerous_opponents, attackable_opponents)
    
    def wandering(self):
        gameState = self.getCurrentObservation()
        dangerous_opponents, _ = self.getObservableOpponents(gameState)
        if len(dangerous_opponents) > 0:
            return False
        
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        q_list = []
        for action in actions:
            q_list.append(self.q_values.get_q_value(gameState, action))

        q_list.sort(reverse=True)
        if len(q_list) > 1 and q_list[0] == q_list[1]:
            print("---------------------------Wandering: Multiple best q_values: {}---------------------------".format(q_list[0]))
            return True
        
        best_action = self.q_values.get_best_action(gameState)
        best_q_value = self.q_values.get_q_value(gameState, best_action)
        choice_history = self.choice_history.get_history()
        if best_action == Actions.reverseDirection(choice_history[-1][0]) and (best_action, best_q_value) == choice_history[-2]:
            print("---------------------------Wandering: Moving forward and backward---------------------------")
            return True
        
        return False
    
    def find_path_to_target(self, gameState, targetPos):
        agentState = gameState.getAgentState(self.index)
        agentPos = agentState.getPosition()
        dangerous_opponents, _ = self.getObservableOpponents(gameState)
        dangerous_opponent_positions = list(map(lambda x: gameState.getAgentState(x).getPosition(), dangerous_opponents))
        locked_positions = set()

        for opponentPos in dangerous_opponent_positions:
            opponent_agent_path_problem = PositionSearchProblem(gameState, agentPos, opponentPos)
            opponent_agent_path = aStarSearch(opponent_agent_path_problem)
            locked_positions |= set(map(lambda x: x[0], opponent_agent_path[ :int(len(opponent_agent_path) / 2) + 1]))

        agent_target_path_problem = PositionSearchProblem(gameState, targetPos, agentPos, locked_states=locked_positions)
        agent_target_path = aStarSearch(agent_target_path_problem)

        return agent_target_path

    def reward(self, gameState, newGameState):
        reward = 0
        agentState = gameState.getAgentState(self.index)
        agentPos = agentState.getPosition()
        newAgentState = newGameState.getAgentState(self.index)
        newAgentPos = newAgentState.getPosition()

        if self.deadState(newGameState):
            return -99999
        elif self.trappedState(newGameState):
            return -9999

        dangerous_opponents, attackable_opponents = self.getObservableOpponents(gameState)
        capsule_positions = self.getCapsules(gameState)
        foodGrid = self.getFood(gameState)
        capture_food = len(foodGrid.asList()) > 2 and agentState.numCarrying < MAX_CAPACITY

        if len(attackable_opponents) > 0:
            attackable_opponent_positions  = list(map(lambda x: gameState.getAgentState(x).getPosition(), attackable_opponents))
            if gameState == self.getCurrentObservation() and newAgentPos in attackable_opponent_positions:
                reward += 150

        targetCapsulePos = None
        if len(dangerous_opponents) > 0 and capture_food:
            for capsulePos in sorted(capsule_positions, key=lambda x: self.getMazeDistance(x, agentPos)):
                agent_capsule_path = self.find_path_to_target(gameState, capsulePos)
                if len(agent_capsule_path) > 0:
                    agent_capsule_distance = len(agent_capsule_path) - 1
                    targetCapsulePos = capsulePos
                    break

            if targetCapsulePos:
                if gameState == self.getCurrentObservation() and newAgentPos == targetCapsulePos:
                    reward += 100
                else:
                    new_agent_capsule_path = self.find_path_to_target(newGameState, targetCapsulePos)
                    new_agent_capsule_distance = len(new_agent_capsule_path) - 1 if len(new_agent_capsule_path) > 0 else math.inf
                    if new_agent_capsule_distance < agent_capsule_distance:
                        reward += 3

        if (len(dangerous_opponents) > 0 and not targetCapsulePos) or not capture_food:
            closest_boundary = self.getClosestPos(gameState, self.boundary)
            agent_boundary_path = self.find_path_to_target(gameState, closest_boundary)
            agent_boundary_distance = len(agent_boundary_path) - 1 if len(agent_boundary_path) > 0 else math.inf
            new_agent_boundary_path = self.find_path_to_target(newGameState, closest_boundary)
            new_agent_boundary_distance = len(new_agent_boundary_path) - 1 if len(new_agent_boundary_path) > 0 else math.inf

            if agentState.isPacman:
                if gameState == self.getCurrentObservation() and newAgentPos == closest_boundary:
                    reward += 100
                elif new_agent_boundary_distance < agent_boundary_distance:
                    reward += 3

        if capture_food and newAgentState.numCarrying > agentState.numCarrying:
            reward += 10

        return reward

    def trappedState(self, gameState):
        agentPos = gameState.getAgentState(self.index).getPosition()
        closest_boundary = self.getClosestPos(gameState, self.boundary)
        agent_boundary_path = self.find_path_to_target(gameState, closest_boundary)
        if len(agent_boundary_path) > 0:
            return False
        
        capsule_positions = self.getCapsules(gameState)
        for capsulePos in sorted(capsule_positions, key=lambda x: self.getMazeDistance(x, agentPos)):
            agent_capsule_path = self.find_path_to_target(gameState, capsulePos)
            if len(agent_capsule_path) > 0:
                return False
        
        return True
    
    def deadState(self, gameState):
        agentPos = gameState.getAgentState(self.index).getPosition()
        if agentPos == gameState.getInitialAgentPosition(self.index):
            return True
        
        return False

    def getNewPos(self, pos, action):
        x, y = pos
        if action == Directions.NORTH:
            y += 1
        elif action == Directions.SOUTH:
            y -= 1
        elif action == Directions.EAST:
            x += 1
        elif action == Directions.WEST:
            x -= 1
        
        return (x, y)
    
    def simulate_opponents_move(self, gameState):
        possibleNewGameStates = []
        possibleOpponentActions = dict()
        agentPos = gameState.getAgentState(self.index).getPosition()
        dangerous_opponents, _ = self.getObservableOpponents(gameState)

        for opponent in dangerous_opponents:
            possibleOpponentActions[opponent] = []
            opponentPos = gameState.getAgentState(opponent).getPosition()
            opponent_agent_distance = self.getMazeDistance(agentPos, opponentPos)
            opponentActions = gameState.getLegalActions(opponent)
            opponentActions.remove(Directions.STOP)

            for opponentAction in opponentActions:
                newOpponentPos = self.getNewPos(opponentPos, opponentAction)
                new_opponent_agent_distance = self.getMazeDistance(agentPos, newOpponentPos)
                if new_opponent_agent_distance < opponent_agent_distance:
                    possibleOpponentActions[opponent].append(opponentAction)

        opponentChoiceCombinations = list(product(*possibleOpponentActions.values()))
        for combination in opponentChoiceCombinations:
            newGameState = gameState
            for opponent, opponentAction in zip(dangerous_opponents, combination):
                newGameState = newGameState.generateSuccessor(opponent, opponentAction)
            possibleNewGameStates.append(newGameState)

        return possibleNewGameStates
    
    def valueIteration(self, gameState, max_iterations=MAX_LEARNING_ITERATIONS):
        states_set = set([gameState])
        states_list = [gameState]
        self.q_values = QFunction()

        for _ in range(max_iterations):
            for state in reversed(states_list):
                if state != gameState and self.deadState(state):
                    continue
                actions = state.getLegalActions(self.index)

                for action in actions:
                    newState = state.generateSuccessor(self.index, action)
                    possibleNewStates = self.simulate_opponents_move(newState)
                    new_q_value = 0

                    if len(possibleNewStates) == 0:
                        if newState not in states_set:
                            states_set.add(newState)
                            states_list.append(newState)
                        new_q_value = self.reward(state, newState) + self.discount_factor * self.q_values.get_best_q(newState)
                    else:
                        avg_possibility = 1 / len(possibleNewStates)
                        for possibleNewState in possibleNewStates:
                            if possibleNewState not in states_set:
                                states_set.add(possibleNewState)
                                states_list.append(possibleNewState)
                            new_q_value += avg_possibility * (self.reward(state, possibleNewState) + self.discount_factor * self.q_values.get_best_q(possibleNewState))

                    self.q_values.update(state, action, new_q_value)
    
        print("Total states: {}".format(len(states_set)))
    
    def chooseAction(self, gameState):
        start_time = time.time()

        agentState = gameState.getAgentState(self.index)
        agentPos = agentState.getPosition()

        dangerous_opponents, attackable_opponents = self.getObservableOpponents(gameState)
        foodGrid = self.getFood(gameState)
        closest_food = self.getClosestPos(gameState,foodGrid.asList())
        capture_food = len(foodGrid.asList()) > 2 and agentState.numCarrying < MAX_CAPACITY

        if len(dangerous_opponents) > 0 or\
                len(attackable_opponents) > 0 or\
                capture_food and self.getMazeDistance(agentPos, closest_food) <= MAX_LEARNING_ITERATIONS:
            print("dangerous opponents: {}, attackable opponents: {}".format(dangerous_opponents, attackable_opponents))
            self.valueIteration(gameState)
            action = self.q_values.get_best_action(gameState)
            q_value = self.q_values.get_q_value(gameState, action)

            if action and not self.wandering():
                print("best action: {}".format(action))
                print("q_value: {}".format(q_value))
                self.choice_history.add(action, q_value)
                print("computation time: {}\n".format(time.time() - start_time))
                return action
              
        if capture_food:
            problem = PositionSearchProblem(gameState, closest_food, agentPos)
            path = aStarSearch(problem)
            self.choice_history.add(path[0][1], None)
            return path[0][1]
        
        else:
            closest_boundary = self.getClosestPos(gameState, self.boundary)
            problem = PositionSearchProblem(gameState, closest_boundary, agentPos)
            path = aStarSearch(problem)
            self.choice_history.add(path[0][1], None)
            return path[0][1]
        