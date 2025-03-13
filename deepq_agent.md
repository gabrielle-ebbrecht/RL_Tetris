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

## train()

Training only occurs when the memory has at least replay_start_size experiences and there are at least batch_size samples available (i.e., batch_size > mem_size), preventing training on insufficient experiences and poor generalization of the learned model. If either of these conditions does not hold, the agent will not learn and will rely on random exploration, thus the model weights will not be updated.

```
if batch_size > self.mem_size:
    ...
if n >= self.replay_start_size and n >= batch_size:
    ...
```

During this time, the agent will still be exploring and can collect experiences. The training function will resume once the agent has enough experiences to meet the requirements for training.

We then sample batch_size experiences from memory (random sampling breaks correlations in sequential data) and use the model to predict q-values for the *next_states* in the batch:

```
batch = random.sample(self.memory, batch_size)

next_states = np.array([x[1] for x in batch])
next_qs = [x[0] for x in self.model.predict(next_states)]
```

Q-values are found via the Bellman equation for Q-learning: 

$$Q(s, a) = r + \gamma \max_{a'} Q(s', a')$$

Where $Q(s, a)$ is the expected future reward for taking action $a$ in state $s$, and $r$ is the immediate reward for action $a$ in state $s$. As mentioned, $\gamma$ determines how we weigh future rewards, and is multiplied with the highest expected reward the agent can get from that state, $max_{a'} Q(s', a')$. The use of $\gamma$ also curbs the reward to prevent indefinite growth.

It is important to note that the Bellman equation is implicitly recursive on $Q(s', a')$, i.e., the "recursion" happens over many episodes of training. This allows $\gamma$ to accumulate over multiple steps, therefore discounting future rewards even more as we look further ahead.

We also compute the target values using the Bellman equation. If the episode is done, the Q value is simply the current reward:

```
for i, (state, _, reward, episode_ended) in enumerate(batch):
    if not episode_ended: new_q = reward + self.discount * next_qs[i] # Partial Q formula
    else: new_q = reward
```

Finally, we fit the model to the inputs and q values, and decay epsilon:

```
self.model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)

if self.epsilon > self.epsilon_min:
    self.epsilon -= self.epsilon_decay
```
