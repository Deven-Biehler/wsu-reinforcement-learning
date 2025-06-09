import numpy as np


class GridWorldMazeEnv:
    def __init__(self, seed=None):
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
        self.state = 0
        self.terminal_state = 6
        # The border map ignores the first column, the first row, and the last row of borders.
        self.border_map = [[0, 1, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 1, 0, 0, 0, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 1, 0, 0, 0, 0, 0, 1],
                           [0, 0, 1, 1, 1, 1, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 1, 1, 1, 1, 1],
                           [0, 1, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 1],
                           [0, 0, 1, 1, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1]]
        self.border_map = np.array(self.border_map)
        self.size_of_maze = self.border_map.shape[1]

    def state_to_border(self, state):
        row, column = np.divmod(state, 8)
        border = np.zeros((4,))  # up, right, down, left
        if row == 0:
            border[0] = 1
        else:
            border[0] = self.border_map[row * 2 - 1, column]
        border[1] = self.border_map[row * 2, column]
        if row == self.size_of_maze - 1:
            border[2] = 1
        else:
            border[2] = self.border_map[row * 2 + 1, column]
        if column == 0:
            border[3] = 1
        else:
            border[3] = self.border_map[row * 2, column - 1]
        return border

    def reset(self):
        self.state = 0
        return self.state

    def state_transition_func(self, state, action):
        assert state in range(64), "Error: The state input is invalid!"
        assert action == 0 or action == 1 or action == 2 or action == 3, "Error: The action input is invalid!"
        next_state = state
        border = self.state_to_border(state)
        if action == 0:
            if border[0] == 0:
                next_state = state - 8
        elif action == 1:
            if border[1] == 0:
                next_state = state + 1
        elif action == 2:
            if border[2] == 0:
                next_state = state + 8
        else:
            if border[3] == 0:
                next_state = state - 1
        return next_state

    def step(self, action):
        assert action == 0 or action == 1 or action == 2 or action == 3, "Error: The action input is invalid!"
        self.state = self.state_transition_func(self.state, action)
        if self.state == self.terminal_state:
            reward = 10.0
            done = True
        else:
            done = False
            if self.rng.random() < 0.5:
                reward = -1.0
            else:
                reward = -2.0
        return self.state, reward, done

