import numpy as np
import gym

from gym import spaces


class FourRooms(gym.Env):

    metadata = {"render.modes": ["human"], "id": "FourRooms"}

    def __init__(self):
        super().__init__()

        self.room = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]

        self.height, self.width = len(self.room), len(self.room[0])

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([1, 1, 1, 1]),
            high=np.array([self.width-1, self.height-1, self.width-1, self.height-1]),
            dtype=np.float32
        )

        self.steps = 0

        self.done = False
        self.start = None
        self.goal = None

        self.actionSpace()
        self.reset()

    def actionSpace(self):
        self.LEFT = 0
        self.RIGHT = 1
        self.UP = 2
        self.DOWN = 3

    def step(self, action):

        x = self.pos[0]
        y = self.pos[1]

        if action == self.LEFT:
            if self.room[x - 1][y] == 0:
                self.pos = [x - 1, y]
        elif action == self.RIGHT:
            if self.room[x + 1][y] == 0:
                self.pos = [x + 1, y]
        elif action == self.UP:
            if self.room[x][y - 1] == 0:
                self.pos = [x, y - 1]
        elif action == self.DOWN:
            if self.room[x][y + 1] == 0:
                self.pos = [x, y + 1]

        rew = 0
        if tuple(self.pos) == tuple(self.goal):
            rew = 10
            self.done = True

        self.steps += 1
            
        if self.steps >= 100:
            self.done = True 

        if self.pos == self.goal:
            self.done = True

        state = [self.pos[0], self.pos[1], self.goal[0], self.goal[1]]
        
        return np.array(state, np.float32), rew, self.done, {}

    def reset(self):

        invalid_start = True
        while invalid_start:
            start = [np.random.randint(1, self.width-1), np.random.randint(1, self.height-1)]
            if self.room[start[0]][start[1]] == 0:
                invalid_start = False
                self.start = start

        invalid_goal = True
        while invalid_goal:
            goal = [np.random.randint(1, self.width-1), np.random.randint(1, self.height-1)]
            if (self.room[goal[0]][goal[1]] == 0) and (start != goal):
                invalid_goal = False
                self.goal = goal

        self.pos = self.start
        self.steps = 0
        self.done = False

        state = [self.pos[0], self.pos[1], self.goal[0], self.goal[1]]

        return np.array(state, np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        pass