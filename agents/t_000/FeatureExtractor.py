from game import Directions

class CaptureFeatureExtractor:

    def __init__(self, agent):
        self.agent = agent
        self.allActions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

    def num_features(self):
        return 1

    def num_actions(self):
        return len(self.allActions)

    def extract_features(self, gameState, action):
        feature_values = []
        agentState = gameState.getAgentState(self.agent.index)
        postion = agentState.getPosition()

        for a in self.allActions:
            if a == action and not self.agent.stop_simulation(gameState):
                #feature_values.append(agentState.numCarrying)

                closest_food = self.agent.getClosestPos(postion, self.agent.getFood(gameState).asList())
                distance_closest_food = self.agent.getMazeDistance(postion, closest_food)
                feature_values.append(distance_closest_food)
                '''
                closest_boundary = self.agent.getClosestPos(postion, self.agent.boundary)
                distance_closest_boundary = self.agent.getMazeDistance(postion, closest_boundary)
                feature_values.append(distance_closest_boundary)
                '''

            else:
                for _ in range(0, self.num_features()):
                    feature_values += [0.0]

        return feature_values
    