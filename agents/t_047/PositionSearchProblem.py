from game import Directions, Actions
import util

def getClosestPos(local_pos,pos_list, distance_function):
        min_length = 9999
        min_pos = None
        for pos in pos_list:
            temp_length = distance_function(local_pos,pos)
            if temp_length < min_length:
                min_length = temp_length
                min_pos = pos
        return min_pos

def getClosestDis(local_pos,pos_list, distance_function):
        min_length = 9999
        min_pos = None
        for pos in pos_list:
            temp_length = distance_function(local_pos,pos)
            if temp_length < min_length:
                min_length = temp_length
        return min_length

def aStarSearch(problem):
        """Search the node that has the lowest combined cost and heuristic first."""
        
        from util import PriorityQueue
        myPQ = PriorityQueue()
        startState = problem.getStartState()
        # print(f"start states {startState}")
        startNode = (startState, '',0, [])
        heuristic = problem._manhattanDistance
        myPQ.push(startNode,heuristic(startState))
        visited = set()
        best_g = dict()

        if problem.isGoalState(startState):
            return [(startState, Directions.STOP)]

        while not myPQ.isEmpty():
            node = myPQ.pop()
            state, action, cost, path = node
            # print(cost)
            # print(f"visited list is {visited}")
            # print(f"best_g list is {best_g}")
            if (not state in visited) or cost < best_g.get(str(state)):
                visited.add(state)
                best_g[str(state)]=cost

                if problem.isGoalState(state):
                    output = []
                    path = path + [(node, action)]
                    del path[0]
                    if len(path) == 0:
                        return output
                     
                    output.append((startState, path[0][1]))
                    for i in range(len(path) - 1):
                        output.append((path[i][0][0], path[i + 1][1]))
                    output.append((state, Directions.STOP))

                    return output
                
                for succ in problem.getSuccessors(state):
                    succState, succAction, succCost = succ
                    newNode = (succState, succAction, cost + succCost, path + [(node, action)])
                    myPQ.push(newNode,heuristic(succState)+cost+succCost)

        return []

class PositionSearchProblem:
    """
    It is the ancestor class for all the search problem class.
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point.
    """

    def __init__(self, gameState, goal, start_pos, costFn = lambda x: 1, locked_states=set()):
        self.walls = gameState.getWalls()
        self.costFn = costFn
        x,y = start_pos
        self.startState = int(x),int(y)
        self.goal_pos = goal
        self.locked_states = locked_states

    def getStartState(self):
      return self.startState

    def isGoalState(self, state):

      return state == self.goal_pos

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty] and (nextx, nexty) not in self.locked_states:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )
        return successors

    def getCostOfActions(self, actions):
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost
      
    def _manhattanDistance(self,pos):
      return util.manhattanDistance(pos,self.goal_pos)
    