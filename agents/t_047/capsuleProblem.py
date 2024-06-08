from game import Directions, Actions
from capture import GameState
from agents.t_047.positionSearchProblem import PositionSearchProblem, getClosestPos, getClosestDis, aStarSearch



class CapsuleProblem:
    def __init__(self, remain, gameState:GameState, boundary, index, food) -> None:
        self.remain = remain
        self.gameState = gameState
        self.boudary = boundary
        self.index = index
        self.food = food
        pass

    
    def getBestAction(self, cur_pos, distance_function):
        closest_boundary = getClosestPos(cur_pos, self.boudary, distance_function)
        closest_boundary_dis = distance_function(cur_pos, closest_boundary)
        if len(self.food) > 2:
            closest_food = getClosestPos(cur_pos, self.food, distance_function)
            closest_food_dis = distance_function(cur_pos, closest_food)
        else:
            closest_food_dis = 10000
        path = []
        if (self.remain - closest_boundary_dis - 2 * closest_food_dis > 0):
            path = aStarSearch(PositionSearchProblem(self.gameState, closest_food, cur_pos))
        else:
            path = aStarSearch(PositionSearchProblem(self.gameState, closest_boundary, cur_pos))
        return path[0] if len(path)>0 else None
    
    