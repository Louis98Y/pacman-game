from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import math
from agents.t_047.MCTNode import MCTNode
from agents.t_047.Qfunction import QFunction
from collections import defaultdict
from agents.t_047.positionSearchProblem import *

##########
# Agents #
##########

class DefendAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

        self.boundary = self.getBoundary(gameState)
        self.discount_factor = 0.1
        self.capture_target = None
        #self.last_observed_opponent_position = None
        self.last_defending_foods = set(self.getFoodYouAreDefending(gameState).asList().copy())
        self.roam_end_points = [self.boundary[0], self.boundary[-1]]
        #self.bottlneckProblem = BottleneckProblem(gameState, self.boundary, self.red, self.getMazeDistance)
        self.roam_target = random.choice(list(self.last_defending_foods))


    def roam(self, gameState):
        position = gameState.getAgentPosition(self.index)

        if position == self.roam_target:
            self.roam_target = random.choice(list(self.last_defending_foods))

        problem = PositionSearchProblem(gameState, self.roam_target, position)
        path = aStarSearch(problem)

        if len(path) == 0:
            return random.choice(gameState.getLegalActions(self.index))
        return path[0][1]
    
    
    def getBoundary(self, gameState):
        boundary_location = []
        height = gameState.data.layout.height
        width = gameState.data.layout.width

        for i in range(height):
            if self.red:
                j = int(width/2)-1
            else:
                j = int(width/2)
            if not gameState.hasWall(j,i):
                boundary_location.append((j,i))

        return boundary_location
    
    
    def create_root_node(self, gameState):
        qfunction = QFunction()

        # self, agentIndex, parent, state, qfunction, reward=0.0, parent_action=None
        return MCTNode(
            agent=self,
            parent=None,
            state=gameState,
            qfunction=qfunction
        )
    

    def chooseAction(self, gameState):
        position = gameState.getAgentPosition(self.index)
        agentState = gameState.getAgentState(self.index)

        # Check if there is any observable opponent
        nearest_opponent = self.get_nearest_opponent(gameState)

        if nearest_opponent != None and not agentState.isPacman:
            self.last_observed_opponent = nearest_opponent
            self.capture_target = nearest_opponent
            root_node = self.create_root_node(gameState)
            action = self.mcts(root_node).get_best_action()
            self.last_defending_foods = set(self.getFoodYouAreDefending(gameState).asList().copy())
            #print("chasing observable opponent")
            return action
        
        defending_foods = set(self.getFoodYouAreDefending(gameState).asList())
        lost_foods = list(self.last_defending_foods.difference(defending_foods))
        if len(lost_foods) > 0:
            nearest_opponent = getClosestPos(position, lost_foods, self.getMazeDistance)
            self.capture_target = getClosestPos(nearest_opponent, defending_foods, self.getMazeDistance)

        if self.capture_target != None and position != self.capture_target:
            problem = PositionSearchProblem(gameState, self.capture_target, position)
            path = aStarSearch(problem)
            action = path[0][1]
            #print("chasing unobervable opponent: {}".format(self.capture_target))
        else:
            self.capture_target = None
            action = self.roam(gameState)
            #print("roaming")
        
        self.last_defending_foods = set(self.getFoodYouAreDefending(gameState).asList().copy())
        return action
    

    def get_opponent_positions(self, gameState):
        opponents = self.getOpponents(gameState)
        opponentPositions = list(map(lambda x: gameState.getAgentPosition(x), opponents))
        return opponentPositions
    

    def get_nearest_opponent(self, gameState):
        opponent_positions = self.get_opponent_positions(gameState)
        my_position = gameState.getAgentPosition(self.index)
        min_distance = 9999
        nearest_opponent = None

        for opponent_position in opponent_positions:
            if opponent_position == None:
                continue
            distance = self.getMazeDistance(my_position, opponent_position)
            if distance < min_distance:
                min_distance = distance
                nearest_opponent = opponent_position

        return nearest_opponent
    

    def reward(self, gameState, action):
        reward = 0
        newGameState = gameState.generateSuccessor(self.index, action)
        newPosition = newGameState.getAgentPosition(self.index)

        if newGameState.getAgentState(self.index).isPacman:
            reward -= 100

        if newPosition == self.capture_target:
            reward += 1000

        return reward
    

    def simulate_action(self, gameState):
        my_position = gameState.getAgentPosition(self.index)
        problem = PositionSearchProblem(gameState, self.capture_target, my_position)
        path = aStarSearch(problem)
        if len(path) > 0:
            return path[0][1]
        else:
            legalActions = gameState.getLegalActions(self.index)
            legalActions.remove(Directions.STOP)
            return random.choice(legalActions)
       
    def simulate(self, node):
        """
        Simulate the game from the current node until the end
        """
        gameState = node.state
        depth = 0
        cumulative_reward = 0.0

        while not self.stop_simulation(gameState) and depth < 30:
            # Choose the action to simulate
            choosedAction = self.simulate_action(gameState)

            # Execute the action
            newGameState = gameState.generateSuccessor(self.index, choosedAction)

            # Get the immediate reward
            reward = self.reward(gameState, choosedAction)
            
            # Discount the reward
            cumulative_reward += pow(self.discount_factor, depth) * reward
            depth += 1

            gameState = newGameState

        return cumulative_reward

    def mcts(self, root_node, timeout=0.8):
        iteration = 10

        for _ in range(iteration):

            # Select
            selected_node = root_node.select()

            # Expand
            if not self.stop_simulation(selected_node.state):
                action = random.choice(list(selected_node.unexpandedActions))
                newGameState = selected_node.state.generateSuccessor(self.index, action)
                reward = self.reward(selected_node.state, action)
                child = selected_node.expand(action, newGameState, reward)
            else:
                continue

            # Simulate
            cumulative_reward = self.simulate(child)
            
            # Back propagate
            selected_node.back_propagate(cumulative_reward, child)

        return root_node

    def stop_simulation(self, gameState):
        return len(self.getFood(gameState).asList()) <= 2 or gameState.isOver()
    