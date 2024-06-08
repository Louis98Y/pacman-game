from game import Directions, Actions
from capture import GameState
from agents.t_047.positionSearchProblem import PositionSearchProblem, getClosestPos, getClosestDis, aStarSearch

import util

class CapturedProblem:
    def __init__(self, gameState:GameState, my_pos, opponent_pos, dest_pos, distance_function) -> None:
        self.gameState = gameState
        self.my_pos = my_pos
        self. opponent_pos = opponent_pos
        self.dest_pos = dest_pos
        self.distance_function = distance_function
        pass

    def avoidHeuristic(self, cur_pos):
        danger_dis = 0
        for opponent in self.opponent_pos:
            danger_dis += util.manhattanDistance(opponent, cur_pos) if util.manhattanDistance(opponent, cur_pos) < 5 else 0
 
        return util.manhattanDistance(self.dest_pos, cur_pos) + danger_dis
    
    def verifyPathSafety(self, path):
        safe = True
        step = 1
        x, y = self.my_pos
        for p in path:
            dx, dy = Actions.directionToVector(p)
            x, y = (x+dx, y+dy)
            for opponent in self.opponent_pos:
                if (self.distance_function(opponent, (x,y)) <= step):
                    safe = False
                    break
            if not safe:
                break
            step+=1
        return safe


    def findSafePath(self):
        path = aStarSearch(PositionSearchProblem(self.gameState, self.dest_pos, self.my_pos))
        if self.verifyPathSafety(path) and len(path) != 0:
            return path[0]
        return None
        
        




 