# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement

Implement the Value Iteration algorithm for the FrozenLake environment to compute the optimal state-value function. Using the calculated state-value function, derive the optimal policy by selecting the action that maximizes the expected future reward for each state.

## Software Requirements

Python 3.x

Gymnasium

NumPy

Matplotlib

Jupyter Notebook / Google Colab / VS Code

## Environment Description

```
env_desc = [
    "FHFF",
    "FHFH",
    "SFFH",
    "HFFG"
]
```

## MDP Representation

The FrozenLake problem is represented as a Markov Decision Process (MDP):

States (S): 16 grid positions in a 4×4 environment.
Actions (A): Left, Down, Right, Up.
Transition Probability P: Probability of reaching a particular next state after taking an action.
Reward (R): +1 for reaching the goal and 0 for other transitions.
Discount Factor (γ): 0.99, which determines the importance of future rewards.
Terminal States: Holes and the goal state.

## Theory

Value Iteration is a dynamic programming method used to find the optimal state-value function in an MDP. It repeatedly updates the value of each state using the Bellman optimality equation:
```
                                  Vk+1​(s)=amax​s′∑​P(s′∣s,a)[R(s,a,s′)+γVk​(s′)]
```

The process continues until the maximum change in the value function becomes smaller than the convergence threshold θ. After obtaining the optimal value function, the optimal policy is extracted by choosing the action with the highest expected value.


## Algorithm


1.Initialize the value function V(s) to zero for all states.

2.For each state, evaluate all possible actions.

3.Calculate the expected value of each action using transition probabilities, rewards, and the discount factor.

4.Update the state value with the maximum action value.

5.Calculate the maximum change between old and new values.

6.Repeat until the change is less than θ.

7.Use the final state-value function to determine the best action for every state.

8.Display the optimal state-value function and optimal policy.


## Python Program

```python

import gymnasium as gym
import numpy as np

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env_desc = [
    "FFFF",
    "FHFH",
    "SFFH",
    "HFFG"
]

env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)

P = env.unwrapped.P
n_states = env.observation_space.n
n_actions = env.action_space.n


# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-8):

    P = env.unwrapped.P
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)
    iteration = 0

    while True:
        delta = 0

        for s in range(n_states):

            action_values = np.zeros(n_actions)

            for a in range(n_actions):
                for prob, next_state, reward, done in P[s][a]:
                    action_values[a] += prob * (
                        reward + gamma * V[next_state]
                    )

            best_action = np.argmax(action_values)
            best_value = action_values[best_action]

            delta = max(delta, abs(best_value - V[s]))

            V[s] = best_value
            policy[s] = best_action

        iteration += 1

        if delta < theta:
            break

    return V, policy, iteration


# -------------------------------------------------
# Run Value Iteration
# -------------------------------------------------

V, policy, iterations = value_iteration(env)


# -------------------------------------------------
# Display Output
# -------------------------------------------------
print("Name: P Keerthana")
print("Register Number: 212223240069")
print("Value Iteration Completed")
print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))

action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

```

## Output

<img width="988" height="302" alt="image" src="https://github.com/user-attachments/assets/2c956894-9969-4232-b591-106d58fa9b4f" />



## Result

The Value Iteration algorithm successfully calculated the optimal state-value function for the 4×4 FrozenLake environment. The optimal policy was then obtained from the calculated state-value function by selecting the action with the highest expected value for each state. The agent can therefore follow the generated policy to reach the goal while avoiding holes as effectively as possible.

## Inference

The modified FrozenLake environment converged in 259 iterations, compared to 324 iterations in the original environment. This shows that changing the start position and rearranging the safe and hole states influenced the convergence speed of the Value Iteration algorithm. The updated environment allowed the algorithm to reach the optimal solution in fewer iterations. As a result, the optimal state-value function and policy were also different, reflecting the new environment layout and the best path to the goal.

