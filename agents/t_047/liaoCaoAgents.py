from datetime import datetime
import random
import time

from capture import GameState
from math import inf

from agents.t_047.extendAgents import ExtendAgents
from agents.t_047.VIProblem import ValueIterationProblem
from agents.t_047.capturedProblem import CapturedProblem
from agents.t_047.capsuleProblem import CapsuleProblem
from agents.t_047.positionSearchProblem import PositionSearchProblem, getClosestDis, getClosestPos, aStarSearch

    
class TestValueIterationAgent(ExtendAgents):
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
        super().registerInitialState(gameState)

        '''
        Your initialization code goes here, if you need any.
        '''
        width = gameState.data.layout.width
        self.boundary_x = int(width/2)-1 if self.red else int(width/2)
        width_range = range(self.boundary_x, width) if self.red else range(self.boundary_x+1)

        self.IV = ValueIterationProblem(gameState, width_range, range(gameState.data.layout.height), gama = 0.99)
        self.captured_panalty = getClosestDis(gameState.getAgentPosition(self.index), self.boundary, self.getMazeDistance)
        self.start_pos = None
        # bottleNecks = BottleneckProblem(gameState, self.boundary, self.red, self.getMazeDistance).findBottleneck(4)
        # self.debugDraw(bottleNecks, [1,0,0])
        

    def selectFunction(self, pos, gameState:GameState)->int:
        x ,y = pos

        oppnent_pos_list = self.getOpponentsPos(gameState)
        for oppnent_pos in oppnent_pos_list:
            if oppnent_pos is not None and oppnent_pos == pos:
                if not self.isDanger(gameState):
                    return 0, True
                return -self.captured_panalty - self.carrying, True
        
        foods = gameState.getBlueFood() if self.red else gameState.getRedFood()
        walls = gameState.getWalls()

        if self.isDanger(gameState):
            return -len([(dx,dy) for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)] if walls[x+dx][y+dy]]), False

        if pos in foods.asList():
            wall_cnt = 0
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                if walls[x+dx][y+dy]:
                    wall_cnt+=1
            
            
            return 1.0 if wall_cnt <3 else 1.5, True
        
        if pos[0] == self.boundary_x:
            if self.isDanger(gameState):
                return self.carrying + self.captured_panalty
            if len(foods.asList()) <=2:
                return_len = 0
            else:
                return_len = getClosestDis(gameState.getAgentPosition(self.index), self.boundary, self.getMazeDistance)
            value = self.carrying - 0.6* return_len
            return value if value > 0 else 0, True
        
        return 0.0, False
    
    def getStartBoundary(self, cur_pos):
        return getClosestPos(cur_pos, self.boundary, self.getMazeDistance)
    
    def getBestAction(self, gameState:GameState):
        # print("debug !!!")
        # self.IV.setLayoutValue(gameState, self.selectFunction)
        cur_pos = gameState.getAgentState(self.index).getPosition()
        if self.start_pos is None:
            self.start_pos = self.getStartBoundary(cur_pos)

        if not self.isPacMan(gameState):
            path  = aStarSearch(PositionSearchProblem(gameState, self.start_pos, cur_pos))
            return path[0] if len(path) > 0 else random.choice(gameState.getLegalActions(self.index))
        
        if self.isDanger(gameState):
            captured_problem = CapturedProblem(gameState, cur_pos, self.getOpponentsPos(gameState), None, self.getMazeDistance)
            if self.carrying:
                for b in self.boundary:
                    captured_problem.dest_pos = b
                    action = captured_problem.findSafePath()
                    if action is not None:
                        print("capture return")
                        return action
                for c in self.getCapsules(gameState):
                    captured_problem.dest_pos = c
                    action = captured_problem.findSafePath()
                    if action is not None:
                        print("capture get capsule")
                        return action
            else:
                for c in self.getCapsules(gameState):
                    captured_problem.dest_pos = c
                    action = captured_problem.findSafePath()
                    if action is not None:
                        print("capture get capsule")
                        return action
                for f in self.getFood(gameState).asList():
                    captured_problem.dest_pos = f
                    action = captured_problem.findSafePath()
                    if action is not None:
                        print("capture find cut in")
                        return action
                
                captured_problem.dest_pos = self.boundary[random.randint(0,len(self.boundary)-1)]
                action = captured_problem.findSafePath()
                if action is not None:
                    print("capture random boundary walking")
                    return action

        
        if self.capsulate:
            if self.carrying > 0:
                action = CapsuleProblem(self.capsulate_remain, gameState, self.boundary, self.index, self.getFood(gameState).asList()).getBestAction(cur_pos, self.getMazeDistance)
                if action is None:
                    print("Capsulate return null")
                else:
                    print("capsulate walk")
                    return action

        self.IV.setLayoutValue(gameState, self.selectFunction)
        delta = 1
        cnt = 0
        start_time = time.time()
        while delta > 0.01:
            cnt+=1
            delta = self.IV.iterateValue()
            if cnt%10000 == 0:
                print(f"cnt {cnt}, delta {delta}")
            # self.debugClear()
        # ma = -inf
        # mi = inf
        # for pos in self.IV.layout.keys():
        #     ma = max(ma,self.IV.layout[pos].values[0])
        #     mi = min(mi,self.IV.layout[pos].values[0])
        # for pos in self.IV.layout.keys():
        #     if ma==mi :
        #         break
        #     a = (self.IV.layout[pos].values[0]-mi)/(ma-mi)
        #     self.debugDraw([pos], [a,0,0])
        action = self.IV.getBestAction(cur_pos)
        
        
            
        # debug
        # self.IV.debugPrintLayout()
        print(f"iterate {cnt} times, consume {time.time()-start_time}, delta {delta}")
        return action
    
    

class DefendBaselineAgent(ExtendAgents):
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
        super().registerInitialState(gameState)

        '''
        Your initialization code goes here, if you need any.
        '''
        width = gameState.data.layout.width
        self.boundary_x = int(width/2)-1 if self.red else int(width/2)
        width_range = range(self.boundary_x, width) if self.red else range(self.boundary_x+1)

        self.IV = ValueIterationProblem(gameState, width_range, range(gameState.data.layout.height), gama = 0.99)
        self.captured_panalty = getClosestDis(gameState.getAgentPosition(self.index), self.boundary, self.getMazeDistance)
        self.start_pos = None
        bottleNecks = BottleneckProblem(gameState, self.boundary, self.red, self.getMazeDistance).findBottleneck(5)
        bnp = BottleneckProblem(gameState, self.boundary, self.red, self.getMazeDistance)
        influence = []
        total_food_number = len(gameState.getRedFood().asList() if self.red else gameState.getBlueFood().asList())
        for i in range(1, len(bottleNecks)+1):
            if len(bnp.influence(bottleNecks[:i])) >= total_food_number -3:
                influence = i 
                print(f"use i = {i}")
                break
        
        self.bottleNecks = bottleNecks[:i]
        self.debugDraw(self.bottleNecks, [1,0,0])

        self.state = "roam"
        self.roam_target_index = 0
        self.roam_target = self.bottleNecks[self.roam_target_index%len(self.bottleNecks)]
        self.chase_target = None
        self.chase_step = 0
    
    def getBestAction(self, gameState:GameState):
        cur_pos = gameState.getAgentState(self.index).getPosition()

        if self.getLatestLostFood() is None:
            self.state = "roam"
        else:
            self.state = "chase"
            self.chase_step += 1
            print(f"LatestLostFood find {self.getLatestLostFood()}, to chase")
            if self.chase_target != self.getLatestLostFood():
                print(f"new LatestLostFood find {self.getLatestLostFood()}, to chase")
                self.chase_target = self.getLatestLostFood()
                self.chase_step = 0
            elif self.chase_step >= 7:
                print("target lose, back to roam")
                self.state = "roam"
        
        observed_opponent = self.getOpponentsPos(gameState)
        if len(observed_opponent) != 0:
            for pos in observed_opponent:
                if (self.red and pos[0] <= self.boundary_x) or (not self.red and pos[0]>=self.boundary_x):
                    print("observed opponent")
                    self.state = "chase"
                    self.chase_target = observed_opponent[0]
                    break
            

        if self.state is "chase":
            if cur_pos == self.chase_target:
                print(f"chase to {self.chase_target} finished")
                self.chase_step = 8
                self.state = "roam"
            else:
                print(f"chase to {self.chase_target}")
                path = aStarSearch(PositionSearchProblem(gameState, self.chase_target, cur_pos))
                return path[0]


        if self.state is "roam":
            if cur_pos == self.roam_target:
                print(f"reach roam target, go to next")
                self.roam_target_index += 1
                self.roam_target = self.bottleNecks[self.roam_target_index%len(self.bottleNecks)]
            print(f"roam to {self.roam_target}")
            path = aStarSearch(PositionSearchProblem(gameState, self.roam_target, cur_pos))
            return path[0]
        



