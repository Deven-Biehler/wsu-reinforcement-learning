# Imagine an agent moving within this $8\times 8$ gridworld, where the grids are numbered from $0$ to $63$, the red lines represent walls, and the green grid (grid $0$) and the blue grid (grid $6$) are the starting point and the destination, respectively. The agent can move up, right, down, or left for each step except that the walls block the agent's path. That is, if there is a wall in the direction that the agent plans to move, the agent will remain in the current cell.
# 
# If the agent arrives at grid $6$ (the destination), the agent will receive a reward of $+10$ and the process will terminate. Otherwise, the agent will receive a reward of -1 with probability 0.5 and a reward of -2 with probability 0.5 for each step (including hitting the wall).
# 
# The agent's goal is to find the optimal policy that maximizes the expected discounted total reward starting from grid $0$ to grid $6$. The discount factor is $0.9$.
# 
# Formulation:
# 
# - *State $s$*:
# 
#     The state is defined as the grid where the agent is located. $s\in\{0,1,...,63\}$.
#     The initial state is $0$ and the terminal state is $6$.
# 
# 
# 
# - *Action $a$*:
# 
#     - $a=0$: the agent plans to move *up*;
#     - $a=1$: the agent plans to move *right*;
#     - $a=2$: the agent plans to move *down*;
#     - $a=3$: the agent plans to move *left*.
# 
# 
# 
# - *Transition*:
#     
#     Examples:
#     - If $s=0$ and $a=0$, then the next state will be $s'=0$ (The agent hits the wall);
#     - If $s=0$ and $a=1$, then the next state will be $s'=1$;
#     - If $s=0$ and $a=2$, then the next state will be $s'=8$;
#     - ...
# 
# 
# 
# - *Random reward $r(s,a)$*:
# 
#     $r(5,1) = 10$, $r(7,3) = 10$, $r(14,0) = 10$. Otherwise, $r(s,a)$ is equal to $-1$ with probability $0.5$ and $-2$ with probability $0.5$.
# 
# 
# - *Objective*: Maximize the expected discounted total reward:
# 
#     $$\mathbb{E} \left[\sum_{t=0}^{\tau - 1} \alpha^t r(s_t,a_t) | s_0=0 \right]$$
# 
#   where the subscript $t$ denotes the time slot, $\tau$ is the time slot when the agent reaches the terminal state $6$ starting from $s_0$, and $\alpha=0.9$ is the discount factor.
# 
# Note that in this problem the state transition is deterministic. Define a determinitic state transition function $f$ such that $s' = f(s,a)$ where $s$ is the current state, $a$ is the current action, and $s'$ is the next state. For example, $f(0,0)=0$, $f(0,1)=1$.
# 
# Let $V^*(s)$ denote the optimal value function, defined by
# 
# $$V^*(s) = \max_{\mu} \mathbb{E} \left[\sum_{t=0}^{\tau - 1} \alpha^t r(s_t,\mu(s_t)) \vert s_0 = s \right]$$
# 

# In[2]:


# Upload the file "gridworld_maze.py"


# In[1]:


# Import packages. Run this cell.

import numpy as np
from gridworld_maze import GridWorldMazeEnv
env = GridWorldMazeEnv(seed=0)


# **(1).** Value Iteration (10 pts)
# 
# Use value iteration to compute the optimal value function $V^*(s)$ for all $s\in\{0,1,...,63\}$.
# 
# You may use the function `env.state_transition_func` for the state transition function $f$.
# For example,
# 
# `next_state = env.state_transition_func(current_state, current_action)`
# 
# Please complete the function `value_iteration()`, which return the optimal value function, a numpy array `V` with size 64. The required precision is $0.01$.

# In[19]:


def value_iteration(): # Question: There is no input, how can I iterate on a non existant vector? is this just to initialize? why is the title misleading then.
    """
    Please use value iteration to compute the optimal value function and store it into the vector V with size 64
    For example, V[0] means the optimal value function for state 0.
    """
    V = np.zeros((64,))
    V_prev = np.zeros((64,))
    n = 0

    V[5] = 10
    V[7] = 10
    V[14] = 10
    while (True):
      n+=1
      for current_state in range(len(V)):
        next_states = []
        next_state_values = []
        for current_action in [0, 1, 2, 3]:
          next_state = env.state_transition_func(current_state, current_action)
          next_states.append(next_state)
          next_state_values.append(V[next_state])
        if current_state == 6:
          V[current_state] = 0
        elif 6 in next_states:
          V[current_state] = 10
        else:
          action = np.argmax(next_state_values)
          next_state = next_states[action]
          V[current_state] = -1.5 + 0.9 * V[next_state]
      if (np.max(np.abs(V - V_prev)) < 0.01):
        break
      V_prev = V.copy()
    return V


# In[20]:


# Sample Test, checking the values of V

V = value_iteration()

# Sample output:
# V[6] = 0
# V[5] = 10

# Sample test
print("Final Vector")
print(V)
assert round(V[6], 1) == round(0.0, 1), "Question (1): The sample test failed. V[6] should be 0."
assert round(V[5], 1) == round(10.0, 1), "Question (1): The sample test failed. V[5] should be 10."



# **(2).** Agent's Policy (5 pts)
# 
# Assume that you have obtained the optimal value function $V^*(s)$ by value iteration. Please implement the agent's policy using $V^*(s)$ by completing the function `policy_v`.
# 
# The input `state` is the current state.
# The input `V_star` is the optimal value function $V^*(s)$.
# The output `action` is the action that the agent plans to take.
# 
# You may use the function `env.state_transition_func` for the state transition function $f$.
# 

# In[ ]:


def policy_v(state, V_star):
    """
    Implement the policy of the agent given the optimal value function
    Args:
        state: the current state of the agent, i.e, the grid where the agent is located.
        V_star: the optimal value function V*, a numpy array with size $64$
    Returns:
        action: the action that the agent plans to take.
                The value of action should be 0, 1, 2, or 3, representing up, right, down, or left, respectively.
    """
    # Simply take the maximum value over all 4 actions for each cell.

    next_states = []
    scores = []
    for i in range(4):
      next_states.append(env.state_transition_func(state, i))
      if next_states[-1] == 6:
        scores.append(10)
      else:
        scores.append(-1.5 + 0.9 + V_star[next_states[-1]])

    action = np.argmax(scores)

    return action


# In[ ]:


# Sample Test, check the output of your function policy_v

# Sample input
V_star = np.zeros((64,))
V_star[0] = -14.05
V_star[1] = -13.94
V_star[9] = -13.82
# Note that the above V_star is just an example for a sanity check of your function policy_v.
# It may or may not be the true optimal value function

state = 1

# Sample output
action = 2  # go down to 9

# Sample test
func_out = policy_v(state, V_star)
print(func_out)

# The autograder will run your policy in the gridworld environment. We will check your total reward averaged over 500 episodes.


# **(3).** Policy Iteration (15 pts)
# 
# Please complete the Python function `policy_iteration` to implement the policy iteration algorithm.
# 
# The output `mu` is the optimal policy obtained from the policy iteration algorithm. `mu` is a numpy array with `mu[s]` being the action to take in state $s$ following the policy. For example, `mu[0]=1` means that the agent will move right if it is in state $0$ following the policy `mu`.
# 
# You may use the function `env.state_transition_func` for the state transition function $f$.
# You may use the function `policy_evaluation` to get the value function of a given policy in the policy evaluation step.

# In[ ]:


#  This function is provided for the policy evaluation step in the policy iteration algorithm
def policy_evaluation(mu):
    """
    Policy evaluation: calculate the value function of a given policy
    Args:
        mu: a given policy, which is a numpy array with size 64.
            Example: mu[s] = 0 or 1 or 2 or 3, which represents the action in state s
    Returns:
        V: the value function of the given policy mu, which is a numpy array with size $64$
    """
    V = np.zeros((64,))
    V_pre = np.zeros((64,))
    eps = 1e-3
    error = 100
    while error >= eps:
        for s in range(64):
            if s == 6:
                V[s] = 0
            else:
                a = mu[s]
                if (s == 5 and a == 1) or (s == 7 and a == 3) or (s == 14 and a == 0):
                    V[s] = 10
                else:
                    V[s] = -1.5 + 0.9 * V_pre[env.state_transition_func(s, a)]
        error = np.max(np.abs(V_pre - V))
        V_pre = V.copy()
    return V


# In[ ]:


def policy_iteration():
    """
    Implement the policy iteration algorithm
    Returns:
        mu: the optimal policy obtained from the policy iteration algorithm.
            mu is a numpy array with size 64.
            mu[s] = 0 or 1 or 2 or 3, which represents the action in state s
    """
    # Initial Policy
    mu = np.zeros((64,))

    # Compute Value Function
    V = policy_evaluation(mu)

    n = 0

    V[5] = 10
    V[7] = 10
    V[14] = 10

    # Compute New Policy

    while (True):
      n+=1
      for current_state in range(len(V)):
        next_states = []
        next_state_values = []
        for current_action in [0, 1, 2, 3]:
          next_state = env.state_transition_func(current_state, current_action)
          next_states.append(next_state)
          next_state_values.append(V[next_state])
        if current_state == 6:
          pass
        elif 6 in next_states:
          mu[current_state] = next_states.index(6)
        else:
          action = np.argmax(next_state_values)
          mu[current_state] = action

      # Compare Policy Change
      V_prev = V.copy()

      V = policy_evaluation(mu)
      
      if (np.max(np.abs(V - V_prev)) < 0.01):
        break

    return mu


# In[ ]:


# Sample Test, check the output of your function policy_iteration

# Sample output
mu_1 = 2  # mu[1] = 2
mu_5 = 1  # mu[5] = 1

# Sample test
func_out = policy_iteration()


# In[ ]:


# The code in this cell is provided for debugging.
# The following code will run your policy in the gridworld environment. The maximum length of each episode is set to be 1000.
# We will print your agent's actions during the last episode and the total reward averaged over 500 episodes.
# You can check whether your policy is optimal by these actions.
policy = policy_iteration()
print("Your actions during the last episode:")
total_reward = 0.0
num_ep = 500
for ep in range(num_ep):
    state = env.reset()
    for step in range(1000):
        action = int(policy[state])
        if ep == num_ep - 1:
            print(action, end=" ")
        next_state, reward, done = env.step(action)
        total_reward = total_reward + (0.9 ** step) * reward
        if done:
            break
        else:
            state = next_state
total_reward = total_reward / num_ep
print("")
print("Your total reward averaged over %d episodes:\n%.3f" % (num_ep, total_reward))

# Sample test


