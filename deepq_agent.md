# DQN Agent Class: Explanation

## __init__()

Replay memory is important in reinforcement learning since consecutive events are highly correlated. Replay memory helps us to avoid overfitting to more recent interactions with the environment by:
- Randomizing the mix of training data
- Storing past experiences so they can be reused during training

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