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

    def __init__(self): # initialize attributes
        return None
    

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