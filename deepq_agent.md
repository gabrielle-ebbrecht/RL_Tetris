# DQN Agent Class: Explanation

## __init__()

Replay memory is important in reinforcement learning since consecutive events are highly correlated. Replay memory helps us avoid overfitting to more recent interactions with the environment by:
- Randomizing the mix of training data
- Storing past experiences so they can be reused during training

The memory size is fixed, so older memories are removed over time when this threshold is reached.

```
self.mem_size = mem_size
self.memory = deque(maxlen=mem_size)
```

The discount factor ($\gamma$) allows us to change how we weigh immediate vs long-term learnings:

$$
G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \gamma^3 R_{t+3} + \dots
$$

where G_t is the total discounted return at time t and R_t is the reward received at t. Smaller values for $\gamma$ (near 0) weigh immediate rewards more highly, whereas values closer to 1 weigh longer-term rewards more equally.

```
# Our discount is chosen as 0.95 for more equal consideration
self.discount = discount
```

The exploration-exploitation tradeoff in DQN determines the probability of the agent choosing a random action instead of selecting the best-known action. Epsilon of 1 indicates pure exploration as we learn. The epsilon decay parameter defines how quickly epsilon decreases over time. With epsilon_stop_episode, which is the episode number at which epsilon reaches its minimum value, we can increase/descrease the rate of decay, or make episilon static if epsilon_stop_episode == 0. We also set a minimum exploration probability to prevent epsilon from reaching zero and leaving no room for exploration.

```
self.epsilon = epsilon
self.epsilon_min = epsilon_min
self.epsilon_decay = (epsilon - epsilon_min) / epsilon_stop_episode if epsilon_stop_episode > 0 else 0
```

## predict_score()

This function returns the expected value, or the *Q-value*, of a given state.

We first reshape the state so it can be properly processed by the neural network:

```
state = np.reshape(state, [1, self.state_size])
```

If a randomly generated number is less than or equal to self.epsilon, we return a random score, introducing randomness into the agent and encouraging exploration. Taking random actions allows the agent to discover strategies that may be better. 

Otherwise, we predict the expected value of the state, and the agent selects the best-known action based on past experience (exploitation). As epsilon decays, there will be a lower probability of exploration in our score prediction function.

```
if random.random() <= self.epsilon: return random.random()
else: return self.model.predict(state, verbose=0)[0]
```

## get_best_states()

Again, if a randomly generated number is less than or equal to the current epsilon value, we will choose a state at random for some exploration. Otherwise, we iterate through the states and decide the best state based on the highest predicted score outcome

## add_to_memory()

This function adds a move to the replay memory for use during training. We append the experience tuple of the current state before taking the action, the state after taking the action, the reward for the action, and whether the current episode has ended. The episode (or this series of interactions) ends when the agent gets a "Game Over". This is called the *terminal condition*.