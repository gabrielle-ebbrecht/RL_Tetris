# DQN Training: Explanation

The training process is intuitive:
1. Initialize necessary parameters for training
2. Create the DQN agent
3. Create the logging directory and TensorBoard logger
4. Start each episode
5. Have the agent choose the best move and play it until it gets Game Over
6. Store the game score

## Parameters
```
episodes = 1500 
episode_step_limit = None 
train_every = 1
epochs = 1
log_every = 50 
render_every = 50 
render_delay = None
```

*episodes* refers to the number of games the agent will play during training. The *episode_step_limit* defines the maximum number of moves in a game; hence, when set to None the game will continue until the agent loses (the blocks collide with the top of the board). We train the model every *train_every* episodes and train the model *epochs* number of times per episode. We log our score stats (min, max, mean) every *log_every* episodes. We also render the full gameplay (as opposed to quickly passing game screens with all moves already made) every *render_every* episodes to help with efficiency while still providing visual feedback.

```
epsilon_stop_episode = 2000 
discount = 0.95
```

*epsilon_stop_episode* sets the episode number where exploration will stop. *discount* is $\gamma$, or the discount factor of future rewards. 


```
mem_size = 1000 
replay_start_size = 1000 
batch_size = 128
```

*mem_size* defines the maximum number of experiences (moves) that can be stored in the replay memory, and we're only able to start training after we have gathered *replay_start_size* experiences. *batch_size* is the size of the batch we sample from memory. 

```
n_neurons = [32, 32, 32]
activations = ['relu', 'relu', 'relu', 'linear']
save_best_model = True
```

The number of neurons in each hidden layer of the neural network is denoted by *n_neurons* and the activation functions used at each layer are defined in *activations*. ReLU activations are used for all hidden layers for sparse activations and efficient training, while a linear activation is used in the output layer to predict Q values, which can be any real number.

Finally, we set *save_best_model* to True in order to overwrite our previous best model with an improved model.

## Visualization

While the training process is fairly self-explanatory, there is additional score plotting after tranining is complete to visualize the gameplay improvement over time.

![My Image](gameplay_improvement.jpg)