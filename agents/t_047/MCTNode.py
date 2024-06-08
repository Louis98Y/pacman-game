import math
import time
import random
from collections import defaultdict
from game import Directions
from agents.t_047.Qfunction import QFunction


class MCTNode:

    # Records the number of times states have been visited
    visits = defaultdict(lambda: 0)

    def __init__(self, agent, parent, state, qfunction, reward=0.0, parent_action=None):
        self.agent = agent
        self.parent = parent
        self.state = state
        self.children = []

        self.unexpandedActions = state.getLegalActions(agentIndex=self.agent.index)
        self.unexpandedActions.remove(Directions.STOP)

        # The Q function used to store state-action values
        self.qfunction = qfunction

        # The immediate reward received for reaching this state, used for backpropagation
        self.reward = reward

        # The action that generated this node
        self.parent_action = parent_action

    def is_fully_expanded(self):
        return len(self.unexpandedActions) == 0
    
    def is_terminal(self):
        return self.agent.stop_simulation(self.state)

    '''Select a node that is not fully expanded'''

    def select(self):
        if not self.is_fully_expanded() or self.is_terminal():
            return self
        else:
            best_uct = -math.inf
            selected_child  = None

            for child in self.children:
                mcts_value = self.qfunction.get_q_value(self.state, child.parent_action)
                ucb_value = 2 * math.sqrt(2 * math.log(MCTNode.visits[self.state]) / MCTNode.visits[(self.state, child.parent_action)])
                uct_value = mcts_value + ucb_value

                if uct_value > best_uct:
                    best_uct = uct_value
                    selected_child = child

            return selected_child.select()


    """ Expand a node if it is not a terminal node (checked in MCTSAgent) """

    def expand(self, action, next_state, reward):
        new_child = MCTNode(self.agent, self, next_state, self.qfunction, reward, action)
        self.children.append(new_child)
        self.unexpandedActions.remove(action)

        return new_child


    """ Backpropogate the reward back to the parent node """

    def back_propagate(self, reward, child):
        action = child.parent_action

        MCTNode.visits[self.state] = MCTNode.visits[self.state] + 1
        MCTNode.visits[(self.state, action)] = MCTNode.visits[(self.state, action)] + 1

        # Update the Q function
        q_value = self.qfunction.get_q_value(self.state, action)
        visits = MCTNode.visits[(self.state, action)]
        # use 1 / visits as learning rate
        delta = (1 / visits) * (reward - q_value)
        new_q_value = q_value + delta
        self.qfunction.update(self.state, action, new_q_value)

        if self.parent != None:
            self.parent.back_propagate(self.reward + reward, self)
    
    def get_best_action(self):
        
        max_q_value = -math.inf
        best_action = None

        for child in self.children:
            action = child.parent_action
            q_value = self.qfunction.get_q_value(self.state, action)
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action

        if best_action == None:
            legalActions = self.state.getLegalActions(agentIndex=self.agent.index)
            legalActions.remove(Directions.STOP)
            best_action = random.choice(legalActions)
          
        return best_action

        '''
        best_action = None
        visits = 0
        for child in self.children:
            if MCTNode.visits[(self.state, child.parent_action)] > visits:
                visits = MCTNode.visits[(self.state, child.parent_action)]
                best_action = child.parent_action

        print("Best action is " + str(best_action) + " with Q visit times " + str(visits))
        return best_action
        '''
        