from maze import MazeEnv
from q_learning import QLearningAgent

# E a training the agent
# E b monitor agent rewards earned
# E c visualize learned policy and trajectory
def train_agent(env, agent, episodes):

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        if episode%100 == 0:
            print(f"Episode {episode+1}: Total Reward: {total_reward}")

def evaluate_agent(env, agent, episodes):
    total_steps = 0

    for _ in range(episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done:
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            env.render()
            steps += 1

            if done and reward == 1:
                total_steps += steps
                break  

    avg_steps_per_episode = total_steps / episodes

    
    print(f"Average Steps per Episode: {avg_steps_per_episode}")

if __name__ == "__main__":

    maze = [
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0]
    ]
#     maze = [
#     [0, 0, 1, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 1, 0, 1, 0],
#     [1, 1, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 1, 0, 1],
#     [0, 1, 1, 0, 0, 0, 1, 0],
#     [0, 1, 0, 0, 1, 0, 0, 0],
#     [1, 0, 1, 0, 0, 1, 1, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0]
# ]
    env = MazeEnv(maze)
    agent = QLearningAgent(env)
    
    train_agent(env, agent, episodes=1000)
    evaluate_agent(env, agent, episodes=1)