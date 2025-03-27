# Hi.
### This is my attempt at learning about deep Q networks (and RL in general) via one of my favorite games.

Note: [This Tetris implementation](https://github.com/nuno-faria/tetris-ai/blob/master/tetris.py) by [nuno-faria](https://github.com/nuno-faria) served as a baseline for this work and were extremely helpful in understanding the game and agent-building requirements.

### Checklist:
1. Create the tetris environment. I should be able to play by myself using functions? - DONE
2. Create the DQN - DONE
3. Train and test
4. test w/ jax as backend
5. MD file with code walkthroughs - DONE

# Documentation (for my reference)
- https://keras-rl.readthedocs.io/en/latest/agents/overview/
- https://www.geeksforgeeks.org/deep-q-learning/
- https://medium.com/@shruti.dhumne/deep-q-network-dqn-90e1a8799871
- https://domino.ai/blog/deep-reinforcement-learning

# Deep Q Networks Overview
Reinforcement learning leverages continuous experience and interaction with the environment rather than learning from existing data in a supervised manner. Learning is enabled by a rewards-based system, i.e., the agent aims to maximize the cumulative reward over time. 

Q learning is a reinforcement learning algorithm that uses a lookup table of Q values for each state-action pair to to find the optimal policy for the agent. Read more on the bellman equation and Q value calculation in the [DQN code walkthrough](https://github.com/gabrielle-ebbrecht/RL_Tetris/blob/main/deepq_agent.md) for this repo.

Because Q tables are impractical in large state spaces, with exponential growth in the size of the table, an alternative method was adopted to approximate Q values: deep neural networks! Instead of mapping a state-action pair to the corresponding Q-value, the NN maps states to action-Q-value pairs and is trained to minimize the difference between the predicted and target Q-values using the [Bellman equation](https://www.datacamp.com/tutorial/bellman-equation-reinforcement-learning).

During training, the agent plays the game, collects experiences and stores them in replay memory. It randomly samples a mini-batch of past experiences and trains the network using a loss function based on the Bellman equation. An epsilon-greedy exploration strategy is used to balance exploration and exploitation (read more on epsilon and the exploration-exploitation tradeoff in the [DQN code walkthrough](https://github.com/gabrielle-ebbrecht/RL_Tetris/blob/main/deepq_agent.md)).

# Codebase Tour

This codebase includes the following files:
- *best.keras* - The most recent optimal model for decision-making
- *deepq_agent.md* - Accompanying explanation for the *deepq_agent.py* code
- *deepq_agent.py* - The DQN Agent class and associated functions
- *logs.py* - Custom tensor board class for logging score stats
- *requirements.txt* - All dependencies for running files in this repo
- *run_model.py* - Code for running the saved best model and testing gameplay
- *scores.npy* - The saved scores from the most recent training run
- *tetris.py* - The Tetris game class and associated functions
- *train_dqn.md* - Accompanying explanation for the *train_dqn.py* code
- *train_dqn.py* - Code for training the DQN agent