from deepq_agent import DQNAgent
from tetris import Tetris
from logs import CustomTensorBoard

from datetime import datetime
from statistics import mean
from tqdm import tqdm


def train_agent():
    env = Tetris()
    episodes = 10 # NOTE: CHANGE TO 3000 FOR PROPER TRAINING
    episode_step_limit = None # None --> infinite moves (steps); game ends when DQN gets Game Over
    epsilon_stop_episode = 2000
    mem_size = 1000 # Maximum moves stored by the agent
    discount = 0.95 # Gamma --> discount factor in the Bellman equation for Q-learning
    batch_size = 128 # Number of past experiences sampled from the replay memory. 128 is common in RL scenarios
    epochs = 1 # We train on the batch only once per training step (training too much on the same batch could lead to overfitting)
    render_every = 50 # Avoid inefficiency in training --> game will visually render every 50 episodes
    render_delay = None # None --> game renders as fast as possible. Useful for debugging purposes
    log_every = 50 # Logs the current stats every 50 episodes --> efficient training, better idea of long-term performance
    replay_start_size = 1000 # Minimum experiences required to start training
    train_every = 1 # Train every episode --> continuous learning rather than batch learning
    n_neurons = [32, 32, 32] # Number of neurons for each activation layer
    activations = ['relu', 'relu', 'relu', 'linear'] # Activation layers
    save_best_model = True # Saves the best model so far at "best.keras"

    agent = DQNAgent(
        env.get_state_size(),
        n_neurons=n_neurons, 
        activations=activations,
        epsilon_stop_episode=epsilon_stop_episode, 
        mem_size=mem_size,
        discount=discount, 
        replay_start_size=replay_start_size
        )

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []
    best_score = 0

    for episode in tqdm(range(episodes)):
        current_state = env.reset_board()
        done = False
        step_count = 0
        render = (render_every and episode % render_every == 0)

        while not done and (not episode_step_limit or step_count < episode_step_limit):
            next_states = {tuple(v):k for k, v in env.get_next_states().items()}
            best_state = agent.get_best_state(next_states.keys())
            best_action = next_states[best_state]

            reward, done = env.play(best_action[0], best_action[1], render=render, render_delay=render_delay)
            
            agent.add_to_memory(current_state, best_state, reward, done)
            current_state = best_state
            step_count += 1

        game_score = env.get_score()
        scores.append(game_score)

        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        if log_every and episode and episode % log_every == 0:
            avg_score, min_score, max_score = mean(scores[-log_every:]), min(scores[-log_every:]), max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score,
                    max_score=max_score)

        if save_best_model and game_score > best_score:
            print(f'Saving new optimal model:\nHigh Score={game_score}\nEpisode={episode})')
            best_score = game_score
            agent.save_model("best.keras")


if __name__ == "__main__":
    train_agent()