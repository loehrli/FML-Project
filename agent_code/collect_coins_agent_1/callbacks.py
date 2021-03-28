import os
import pickle
import random
import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.Q_matrices = np.zeros((17,17,17,17,4))
        self.R = np.full((17, 17), -1.)
        self.Q = np.full((17, 17, 4), 0.)
        self.R[0,:] = -100.
        self.R[-1:,:] = -100.
        self.R[:,0] = -100.
        self.R[:, -1] = -100.
        for row in range(2,15):
            for column in range(2, 15):
                if (column % 2 == 0) and (row % 2 == 0):
                    self.R[row, column] = -100.
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
            self.Q_matrices = self.model
            #best_moves = np.argmax(self.Q, axis=2)
            #worst_moves = np.argmin(self.Q, axis=2)
            #print(best_moves)
    self.i = 0
    self.coin_x = 0
    self.coin_y = 0

def get_next_action(self, current_row_index, current_column_index, epsilon):
    #print(self.Q[current_column_index, current_row_index])
    if np.random.random() < epsilon:
        ind = np.argmax(self.Q[current_column_index, current_row_index])
        #print(ind, ACTIONS[ind])
        return np.argmax(self.Q[current_column_index, current_row_index])
    else:
        return np.random.randint(4)


def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index, new_column_index = current_row_index, current_column_index
    action = ACTIONS[action_index]
    #TO DO: check if move valid & if coordinate system correct
    if action == 'UP':
        new_row_index -= 1
    elif action == 'RIGHT':
        new_column_index += 1
    elif action == 'DOWN':
        new_row_index += 1
    elif action == 'LEFT':
        new_column_index -= 1
    return new_row_index, new_column_index


def get_shortest_path(self, nbr_coins, start_row_index, start_column_index):
    #TO DO: check if position is invalid
    current_row_index, current_column_index = start_row_index, start_column_index
    shortest_path = list()
    while self.nbr_coins > 0: # check if terminal state
        action_index = get_next_action(current_row_index, current_column_index, 1)
        current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
        shortest_path.append(action_index)
    return shortest_path

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """ 

    #random_prob = .14
    if self.train: #and random.random() < random_prob:

        for coin in game_state['coins']:
            self.R[coin[1], coin[0]] = 1000.
        
        self.coin_x, self.coin_y = game_state['coins'][0]
        self.i += 1

        #print(self.coin_x, self.coin_y)
        # self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        next_action = get_next_action(self, game_state['self'][3][0], game_state['self'][3][1], 0.7)
        return ACTIONS[next_action]

    #print(game_state['self'][3])
    self.logger.debug("Querying model for action.")

    prev_x, prev_y = self.coin_x, self.coin_y
    if len(game_state['coins']) != 0:
        self.coin_x, self.coin_y = game_state['coins'][0]
    else:
        return 'WAIT'
    self.Q = self.Q_matrices[self.coin_y, self.coin_x]
    if prev_x != self.coin_x and prev_y != self.coin_y:
        print('chasing coin:', self.coin_y, self.coin_x)
        print('Q_matrix:', self.Q.argmax(axis=2))
    self.i += 1
    next_action = get_next_action(self, game_state['self'][3][0], game_state['self'][3][1], 1)
    coins = set(game_state['coins'])
    position = game_state['self'][3]
    #if position in coins:
    #    print('crazy coin collected')
    return ACTIONS[next_action]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)
