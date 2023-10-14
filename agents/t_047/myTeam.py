# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import math
from agents.t_047.MCTNode import MCTNode
from agents.t_047.Qfunction import QFunction
from collections import defaultdict
from agents.t_047.FeatureExtractor import CaptureFeatureExtractor
from agents.t_047.PositionSearchProblem import PositionSearchProblem


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DingyuanAgent1', second = 'DingyuanAgent2'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class MCTSAgent(CaptureAgent):
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
        self.last_action = None

    def getClosestPos(self, gameState, pos_list):
        min_length = 9999
        min_pos = None
        my_local_state = gameState.getAgentState(self.index)
        my_pos = my_local_state.getPosition()
        
        for pos in pos_list:
            temp_length = self.getMazeDistance(my_pos,pos)
            if temp_length < min_length:
                min_length = temp_length
                min_pos = pos

        return min_pos
    
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
            qfunction=qfunction,
            parent_action=self.last_action
        )
    
    def has_observable_opponents(self, gameState):
        opponents = self.getOpponents(gameState)
        for opponentIndex in opponents:
            opponentPosition = gameState.getAgentPosition(opponentIndex)
            if opponentPosition != None:
                return True
            
        return False

    def chooseAction(self, gameState):
        if self.has_observable_opponents(gameState):
            root_node = self.create_root_node(gameState)
            action = self.mcts(root_node).get_best_action()
        else:
            action = self.normal_action(gameState)
        
        return action
    
    def keep_eating(self, gameState):
        return len(self.getFood(gameState).asList()) > 2

    def get_opponent_positions(self, gameState):
        opponents = self.getOpponents(gameState)
        opponent_positions = list(map(lambda x: gameState.getAgentPosition(x), opponents))
        return opponent_positions

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
        agentState = gameState.getAgentState(self.index)
        position = agentState.getPosition()

        newGameState = gameState.generateSuccessor(self.index, action)
        newAgentState = newGameState.getAgentState(self.index)
        newPosition = newAgentState.getPosition()

        opponents = self.getOpponents(gameState)
        opponent_positions = list(map(lambda x: gameState.getAgentPosition(x), opponents))

        reward = 0

        capsules = self.getCapsules(gameState)
        if len(capsules) > 0 and newPosition in capsules:
            reward += 10000

        for opponent_position in opponent_positions:
            if opponent_position == None:
                continue
            
            if newPosition == opponent_position:
                if newAgentState.isPacman:
                    reward -= 10000
                else:
                    reward += 10000
                    
        return reward


    def normal_action(self, gameState):
        if not self.keep_eating(gameState):
            target = self.getClosestPos(gameState, self.boundary)
        else:
            foodGrid = self.getFood(gameState)
            target = self.getClosestPos(gameState, foodGrid.asList())
        
        problem = PositionSearchProblem(gameState, target, self.index)
        path  = self.aStarSearch(problem)
        
        if path == []:
            actions = gameState.getLegalActions(self.index)
            return random.choice(actions)
        else:
            return path[0]
        

    def aStarSearch(self, problem):
        """Search the node that has the lowest combined cost and heuristic first."""
        
        from util import PriorityQueue
        myPQ = util.PriorityQueue()
        startState = problem.getStartState()
        startNode = (startState, '',0, [])
        heuristic = problem._manhattanDistance
        myPQ.push(startNode,heuristic(startState))
        visited = set()
        best_g = dict()

        while not myPQ.isEmpty():
            node = myPQ.pop()
            state, action, cost, path = node
            
            if (not state in visited) or cost < best_g.get(str(state)):
                visited.add(state)
                best_g[str(state)]=cost
                if problem.isGoalState(state):
                    path = path + [(state, action)]
                    actions = [action[1] for action in path]
                    del actions[0]
                    return actions
                for succ in problem.getSuccessors(state):
                    succState, succAction, succCost = succ
                    newNode = (succState, succAction, cost + succCost, path + [(node, action)])
                    myPQ.push(newNode,heuristic(succState)+cost+succCost)
        return []
    
       
    def simulate(self, node):
        """
        Simulate the game from the current node until the end
        """
        gameState = node.state
        depth = 0
        cumulative_reward = 0.0

        while not gameState.isOver() and depth < 10:

            # get actions available
            legalActions = gameState.getLegalActions(self.index)

            # remove stop
            #legalActions.remove(Directions.STOP)

            # remove reversed action
            '''
            reversed_direction = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            if reversed_direction in legalActions and len(legalActions) > 1:
                legalActions.remove(reversed_direction)
            '''

            #choosedAction = self.normal_action(gameState)
            choosedAction = random.choice(legalActions)

            # Execute the action
            newGameState = gameState.generateSuccessor(self.index, choosedAction)

            # Get the immediate reward
            reward = self.reward(gameState, choosedAction)
            
            # Discount the reward
            cumulative_reward += pow(self.discount_factor, depth) * reward
            depth += 1

            gameState = newGameState

        #print(gameState.__str__())

        return cumulative_reward

    def mcts(self, root_node, timeout=0.8):

        start_time = time.time()
        current_time = time.time()
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

            current_time = time.time()

        return root_node

    def stop_simulation(self, gameState):
        return len(self.getFood(gameState).asList()) <= 2
    

class DingyuanAgent1(MCTSAgent):
    pass

class DingyuanAgent2(MCTSAgent):
    pass
