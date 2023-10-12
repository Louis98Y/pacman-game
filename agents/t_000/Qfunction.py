import math
from collections import defaultdict

class QFunction:
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(lambda: 0.0))    # Dictionary to store state-action values

    def update(self, state, action, delta):
        self.q_table[state][action] += delta

    def get_q_value(self, state, action):
        return self.q_table[state][action]
    