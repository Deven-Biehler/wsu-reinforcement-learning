"""
Some parameters
"""
state_size = 24  # state dimension
action_size = 4  # action dimension
fc_units = 256  # number of neurons in one fully connected hidden layer
action_upper_bound = 1  # action space upper bound
action_lower_bound = -1  # action space lower bound


"""
Structure of Actor Network.
"""
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_action = action_upper_bound
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units)
        self.fc3 = nn.Linear(fc_units, action_size)

    def forward(self, state):
        """
        Build an actor (policy) network that maps states -> actions.
        Args:
            state: torch.Tensor with shape (batch_size, state_size)
        Returns:
            action: torch.Tensor with shape (batch_size, action_size)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.max_action
        return action


"""
Structure of Critic Network.
"""
class CriticQ(nn.Module):
    def __init__(self):
        """
        Args:
            state_size: state dimension
            action_size: action dimension
            fc_units: number of neurons in one fully connected hidden layer
        """
        super().__init__()

        # Q-network 1 architecture
        self.l1 = nn.Linear(state_size + action_size, fc_units)
        self.l2 = nn.Linear(fc_units, fc_units)
        self.l3 = nn.Linear(fc_units, 1)

        # Q-network 2 architecture
        self.l4 = nn.Linear(state_size + action_size, fc_units)
        self.l5 = nn.Linear(fc_units, fc_units)
        self.l6 = nn.Linear(fc_units, 1)

    def forward(self, state, action):
        """
        Build a critic (value) network that maps state-action pairs -> Q-values.
        Args:
            state: torch.Tensor with shape (batch_size, state_size)
            action: torch.Tensor with shape (batch_size, action_size)
        Returns:
            Q_value_1: torch.Tensor with shape (batch_size, 1)
            Q_value_2: torch.Tensor with shape (batch_size, 1)
        """
        state_action = torch.cat([state, action], 1)

        x1 = F.relu(self.l1(state_action))
        x1 = F.relu(self.l2(x1))
        Q_value_1 = self.l3(x1)

        x2 = F.relu(self.l4(state_action))
        x2 = F.relu(self.l5(x2))
        Q_value_2 = self.l6(x2)

        return Q_value_1, Q_value_2
    


# The following code is provided for the training of your agent in the 'BipedalWalker-v3' gym environment.
gym.logger.set_level(40)
env = gym.make('BipedalWalker-v3')
_ = env.reset()
env.reset(seed=0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


timesteps_count = 0  # Counting the time steps
max_steps = 1600  # Maximum time steps for one episode
ep_reward_list = deque(maxlen=50)
avg_reward = -9999
agent = TD3()

for ep in range(600):
    state, info = env.reset()
    episodic_reward = 0
    timestep_for_cur_episode = 0

    for st in range(max_steps):
        # Select action according to policy
        action = agent.policy(state)

        # Recieve state and reward from environment.
        next_state, reward, done, truncated, info = env.step(action)
        episodic_reward += reward

        # Send the experience to the agent and train the agent
        agent.train(timesteps_count, timestep_for_cur_episode, state, action, reward, next_state, done)

        timestep_for_cur_episode += 1
        timesteps_count += 1

        # End this episode when `done` is True
        if done:
            break
        state = next_state

    ep_reward_list.append(episodic_reward)
    print('Ep. {}, Ep.Timesteps {}, Episode Reward: {:.2f}'.format(ep + 1, timestep_for_cur_episode, episodic_reward), end='')

    if len(ep_reward_list) == 50:
        # Mean of last 50 episodes
        avg_reward = sum(ep_reward_list) / 50
        print(', Moving Average Reward: {:.2f}'.format(avg_reward))
    else:
        print('')

print('Average reward over 50 episodes: ', avg_reward)
env.close()




