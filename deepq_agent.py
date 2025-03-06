from collections import deque

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.python.keras.initializers import HeNormal
from tensorflow.python.keras.regularizers import l2


# What kinds of functions do I need here?
    # make a choice
    # get the best state

# In general the model needs
    # some objective function

class DQN_Agent():

    def __init__(self, state_size, mem_size=10000, discount=0.95,
             epsilon=1.0, epsilon_min=0.0, epsilon_stop_episode=0,
             n_neurons=[32, 32], activations=['relu', 'relu', 'linear'],
             loss='mse', optimizer='adam', replay_start_size=None, modelFile=None):

        # Check inputs
        if len(activations) != len(n_neurons) + 1: # Why don't we just... hard code n_neurons to match?
            raise ValueError(f"Expected activations list of length {len(n_neurons) + 1}, got {len(activations)}")
        if mem_size <= 0:
            raise ValueError("Memory size must be greater than 0.")
        if replay_start_size is not None and replay_start_size > mem_size:
            raise ValueError("replay_start_size must be less than or equal to mem_size.")

        # Environment & state info
        self.state_size = state_size

        # Replay memory
        self.mem_size = mem_size
        self.memory = deque(maxlen=mem_size)
        self.replay_start_size = replay_start_size if replay_start_size is not None else mem_size // 2

        # Discount factor
        self.discount = discount

        # Exploration (epsilon-greedy strategy)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (epsilon - epsilon_min) / epsilon_stop_episode if epsilon_stop_episode > 0 else 0

        # NN architecture
        self.n_neurons = n_neurons
        self.activations = activations
        self.loss = loss
        self.optimizer = optimizer

        # Load the model
        if modelFile:
            self.model = load_model(modelFile) # If we've already trained
        else:
            self.model = self._build_model()

    

    def build_model(self):
        '''
        Features:
        - L2 Regularization: prevent overfitting. May not be needed for this case. We can compare w/ and w/out
        - Batch normalization: re-center and re-scale activations to improve training stability and speed
        - Dropout: another regulization layer to boost robustness... Also may not be needed
        '''

        model = Sequential()
        model.add(Dense(self.n_neurons[0], input_shape=(self.state_size,), activation=self.activations[0], 
                        kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())

        for i in range(1, len(self.n_neurons)): # Hidden layers
            model.add(Dense(self.n_neurons[i], activation=self.activations[i]))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

        model.add(Dense(1, activation=self.activations[-1]))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model


    def update_model(): 
        return None
    
    def act(): # make a choice
        return None
    
    def predict_score(): # predict the score for an action
        return None
    
    def train(): # more general function to aggregate the others
        return None