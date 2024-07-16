import gym
from gym import spaces

# A a,b environment and observations
# B a custom maze environment with wall, start and stop
# B b functions to initialize, reset and render the maze
# C a actions: move up, down, left, right
# C b observations: current position, proximity to walls or goal
class MazeEnv(gym.Env):

    def __init__(self, maze):
        super(MazeEnv, self).__init__()
        
        self.maze = maze
        self.start = (0, 0)
        self.goal = (len(maze)-1, len(maze[0])-1)
        self.current_position = self.start
        
        self.action_space = spaces.Discrete(4)  
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(maze), len(maze[0])), dtype=int)

    def reset(self):
        self.current_position = self.start
        return self.current_position

    def step(self, action):
        x, y = self.current_position
        
        if action == 0:  
            new_position = (x-1, y)
        elif action == 1: 
            new_position = (x+1, y)
        elif action == 2:  
            new_position = (x, y-1)
        elif action == 3:  
            new_position = (x, y+1)
        
        if self._is_valid_position(new_position):
            self.current_position = new_position
        
        done = self.current_position == self.goal
        reward = 1 if done else -0.1
        
        return self.current_position, reward, done, {}

    def _is_valid_position(self, position):
        x, y = position
        if 0 <= x < len(self.maze) and 0 <= y < len(self.maze[0]) and self.maze[x][y] == 0:
            return True
        return False

    def render(self, mode='human'):
        for i in range(len(self.maze)):
            for j in range(len(self.maze[0])):
                if self.current_position == (i, j):
                    print("A", end=" ")
                elif self.maze[i][j] == 1:
                    print("X", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()
