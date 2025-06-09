# Import packages. Run this cell.
import numpy as np
from gridworld_maze import GridWorldMazeEnv
env = GridWorldMazeEnv(seed=0)

class QLearningAgent:

    def __init__(self, seed=None):
        # The following are recommended hyper-parameters.

        # Initial learning rate: 0.1
        # Learning rate decay for each episode: 0.998
        # Minimum learning rate: 0.001
        # Initial epsilon for exploration: 0.5
        # Epsilon decay for each episode: 0.99

        self.q_table = np.zeros((64, 4))  # The Q table.
        self.learning_rate = 0.1  # Learning rate.
        self.learning_rate_decay = 0.998  # You may decay the learning rate as the training proceeds.
        self.min_learning_rate = 0.001
        self.epsilon = 0.5  # For the epsilon-greedy exploration.
        self.epsilon_decay = 0.99  # You may decay the epsilon as the training proceeds.
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def select_action(self, state):
        """
        This function returns an action for the agent to take.
        Args:
            state: the state in the current step
        Returns:
            action: the action that the agent plans to take in the current step
        """

        # Please complete codes for choosing an action given the current state
        """
        Hint: You may use epsilon-greedy for exploration.
        With probability self.epsilon, choose an action uniformly at random;
        Otherwise, choose a greedy action based on self.q_table.
        """
        ### BEGIN SOLUTION
        # Find the adjecent cell with the maximum Q value
        possible_next_state_values = self.q_table[state]

        # Choose actions with epsilon-greedy
        action = np.argmax(possible_next_state_values)
        ### END SOLUTION

        return action

    def train(self, cur_state, cur_action, reward, next_state, done):
        """
        This function is used for the update of the Q table
        Args:
            - cur_state: the current state
            - cur_action: the current action
            - reward: the reward received
            - next_state: the next state observed
            - `done=1` means that the agent reaches the terminal state (`next_state=6`) and the episode terminates;
              `done=0` means that the current episode does not terminate;
              `done=-1` means that the current episode reaches the maximum length and terminates.
              We set the maximum length of each episode to be 1000.
        """

        # Please complete codes for updating the Q table self.q_table
        """
        Hint: Consider two cases, next_state == 6 and next_state != 6
              You may use self.learning_rate as the learning rate
        """
        ### BEGIN SOLUTION
        self.q_table[cur_state, cur_action] += self.learning_rate * (reward + (self.epsilon_decay) * np.max(self.q_table[next_state]) - self.q_table[cur_state, cur_action])
        ### END SOLUTION

        # Update epsilon and learning rate
        if done != 0:
            self.learning_rate = self.learning_rate * self.learning_rate_decay
            if self.learning_rate < self.min_learning_rate:
                self.learning_rate = self.min_learning_rate
            self.epsilon = self.epsilon * self.epsilon_decay


# The code in this cell is provided for debugging and is also a sample test.
# The following code will train and run your agent in the gridworld environment for 2000 episodes.
# The maximum length of each episode is set to be 1000.
# We will print your agent's actions during the last episode and the total reward averaged over the last 500 episodes.
# You can check whether your policy is optimal by these actions.
np.random.seed(0)
agent = QLearningAgent(seed=0)
print("Your actions during the last episode:")
total_reward = 0.0
num_ep = 2000
rewards_q_learning = np.zeros((num_ep,))
for ep in range(num_ep):
    state = env.reset()
    for step in range(1000):
        action = int(agent.select_action(state))
        if ep == num_ep - 1:
            print(action, end=" ")
        next_state, reward, done = env.step(action)

        rewards_q_learning[ep] = rewards_q_learning[ep] + reward

        if done:
            agent.train(state, action, reward, next_state, 1)
        elif step == 999:
            agent.train(state, action, reward, next_state, -1)
        else:
            agent.train(state, action, reward, next_state, 0)

        if num_ep - ep <= 500:
            total_reward = total_reward + (0.9 ** step) * reward

        if done:
            break
        else:
            state = next_state

total_reward = total_reward / 500
print("")
print("Your total reward averaged over the last %d episodes:\n%.3f" % (500, total_reward))

# Sample test
# Check your total reward averaged over the last 500 episodes
assert total_reward >= -14.2, "Sample test, average total reward is less than -14.2."
