import random
from captureAgents import CaptureAgent
from capture import GameState
from game import Directions, Actions

class ExtendAgents(CaptureAgent):
    def registerInitialState(self, gameState:GameState):
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
        self.boundary = self.getBoundary(gameState)
        self.carrying = 0
        self.capsulate = False
        self.capsulate_remain = -1
        self.protectFood = set(self.getFoodYouAreDefending(gameState).asList())
        self.lostFood = []
        print(self.protectFood)

    def getBestAction(self, gameState:GameState):
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)
    
    def chooseAction(self, gameState):
        action = self.getBestAction(gameState)
        self.step(gameState, action)
        return action
    
    def getBoundary(self,gameState:GameState):
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

    def getOpponentsPos(self, gameState:GameState):
        opponent_indices = self.getOpponents(gameState)
        oppnent_pos_list = []
        for index in opponent_indices:
            oppnent_pos = gameState.getAgentPosition(index)
            if oppnent_pos is not None:
                oppnent_pos_list.append(oppnent_pos)
        
        return oppnent_pos_list

    def isPacMan(self, gameState):
        if self.red:
            return gameState.getAgentState(self.index).getPosition()[0] >= self.boundary_x
        return gameState.getAgentState(self.index).getPosition()[0] <= self.boundary_x

    def isDanger(self, gameState, opponent_close_dis = 5):
        opponent_pos = self.getOpponentsPos(gameState)
        cur_pos = gameState.getAgentState(self.index).getPosition()
        opponent_close = False
        for op in opponent_pos:
            if self.getMazeDistance(cur_pos, op) <=opponent_close_dis:
                opponent_close = True
        return self.isPacMan(gameState) and opponent_close and self.capsulate == False
    
    def step(self, gameState:GameState, action):
        dx,dy = Actions.directionToVector(action)
        x,y = gameState.getAgentState(self.index).getPosition()
        new_x,new_y = int(x+dx),int(y+dy)
        if self.getFood(gameState)[new_x][new_y]:
            self.carrying +=1
        elif (new_x,new_y) in self.boundary:
            self.carrying = 0
        
        if self.capsulate_remain > 0:
            self.capsulate_remain -= 1
            if self.capsulate_remain <= 0:
                self.capsulate = False
                
        if (new_x,new_y) in self.getCapsules(gameState):
            self.capsulate = True
            self.capsulate_remain = 40
        
        last = self.protectFood.copy()
        self.protectFood = set(self.getFoodYouAreDefending(gameState).asList())
        lose = last.difference(self.protectFood)
        if len(lose) != 0:
            print(f"food {lose} is eaten")
            self.lostFood += list(lose)
    
    def getLatestLostFood(self):
        if len(self.lostFood) == 0:
            return None
        return self.lostFood[-1]
    


        


