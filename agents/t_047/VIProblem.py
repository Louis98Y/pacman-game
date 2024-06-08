from game import Directions, Actions
from capture import GameState
class ValueIterationLayoutNode:
    def __init__(self, score = 0) -> None:
        self.neighbors = dict()
        self.values = [score,0]
        self.value_index = 0
        self.gama = 1
        self.reward = score
        self.best_action = Directions.STOP
        self.special = False # if special, not update
    
    def getValueIndex(self):
        return (self.value_index)%2
    
    def getLastValueIndex(self):
        return (self.value_index+1)%2
    
    def bellmanUpdate(self):
        
        value_list = [(self.reward + self.gama*value.values[self.getValueIndex()], key) for key, value in self.neighbors.items()]
        value_list = sorted(value_list, key=lambda x:x[0], reverse=True)
        if (len(value_list) == 0):
            print(f"this node has no neighbor")
            return 0
        best_action = value_list[0][1]
        n = len(value_list)
        best_value = (10-n)/10.0*value_list[0][0]
        for i in range(1,n):
            best_value += 1/10.0*value_list[i][0]
        # best_value, best_action = max(value_list)
        self.best_action = best_action
        # if self.special:
        #     # only calculate action, not update value
        #     return 0
        self.value_index+=1
        self.values[self.getValueIndex()] = best_value
        return abs(self.values[self.getValueIndex()] - self.values[self.getLastValueIndex()])
        

class ValueIterationProblem:
    def __init__(self, gameState:GameState, width_range, height_range, gama = 1) -> None:
        # self.layout = dict()
        self.height = gameState.data.layout.height
        self.width = gameState.data.layout.width
        walls = gameState.getWalls()
        # self.width = gameState.
        
        self.layout = dict(
            map(
                lambda pos:(pos, ValueIterationLayoutNode()), 
                [
                    (x,y) 
                    for x in width_range 
                    for y in height_range 
                    if not walls[x][y]
                ]
            )
        )
        
        for pos, node in self.layout.items():
            x, y = pos
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                dx, dy = Actions.directionToVector(action)
                node.gama = gama
                if (x+dx, y+dy) in self.layout:
                    node.neighbors[action] = self.layout[(x+dx, y+dy)]

    def debugPrintLayout(self):
        debug_info = ""
        for x in range(self.width+1):
            debug_info += f"{x}\t\t"
        debug_info += "\n"

        cnt = 1
        for x in range(self.height-2, 0, -1):
            # now tuning debug info , edge is not show 
            debug_info += f"{cnt}\t"
            # print(f"{cnt}\t", end="")
            cnt += 1
            for y in range(1, self.width+1):
                p = self.layout[(y,x)].values[0] if (y,x) in self.layout else 0
                debug_info += "%.4f"%p + "\t"
                # print("%.4f"%p, end="\t")
            debug_info += "\n"
        with open("agents/t_047/debugoutput.txt", "w") as f:
            f.write(debug_info)
    

    def setLayoutValue(self, gameState, select_function):
        for pos, node in self.layout.items():
            node.values[0], node.special = select_function(pos, gameState)
            node.value_index = 0
            node.reward = 0
            if node.special:
                # node.values[1] = node.values[0]
                node.reward = node.values[0]
    
    def iterateValue(self):
        delta = 0
        for pos, node in self.layout.items():
            delta = max(delta, node.bellmanUpdate())
        return delta
        
    def getBestAction(self, pos):
        if pos not in self.layout:
            print("pos not in layout")
            return Directions.STOP
        return self.layout[pos].best_action