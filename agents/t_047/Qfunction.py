import math
from collections import defaultdict

class QFunction:
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(lambda: 0.0))    # Dictionary to store state-action values

    def update(self, state, action, value):
        self.q_table[state][action] = value

    def get_q_value(self, state, action):
        return self.q_table[state][action]
    
    def get_best_action(self, state):
        best_action = None
        best_q_value = -math.inf

        for action, q_value in self.q_table[state].items():
            if q_value > best_q_value:
                best_action = action
                best_q_value = q_value

        return best_action
    
    def get_best_q(self, state):
        if state not in self.q_table:
            return 0
        best_q_value = -math.inf

        for _, q_value in self.q_table[state].items():
            if q_value > best_q_value:
                best_q_value = q_value

        return best_q_value
    