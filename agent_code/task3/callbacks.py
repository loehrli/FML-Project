import os
import pickle
import random

import math
import numpy as np

import torch


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']
n_actions = 6
steps_done = 0
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    global Q_matrices
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        #with open("q_matrices.pt") as file:
            #self.q_matrices = pickle.load(file)
        #self.model = weights / weights.sum()
        with open("q_matrices.pt", "rb") as file:
            self.model = pickle.load(file)
            Q_matrices = self.model
        with open("backup.pt", "rb") as file:
            self.model = pickle.load(file)
            self.global_policy_net, self.global_target_net = self.model
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
            self.global_policy_net, self.global_target_net = self.model
        with open("q_matrices.pt", "rb") as file:
            self.model = pickle.load(file)
            Q_matrices = self.model

performed_actions = 0

def select_action(self, state):
    global steps_done
    EPS_START = 0.9
    EPS_END = 0.1
    EPS_DECAY = 10000
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    #sample = 1
    eps_threshold = 0.0

    print(eps_threshold)

    global performed_actions
    if performed_actions == 10:
        performed_actions = 0
        #return torch.tensor([[4]])  # Drop bomb if 10 moves without dropping bomb

    performed_actions += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = self.global_policy_net(state_to_features(state)).max(1)[1].view(1, 1)
            #print('trained:', action)
            
            if action == torch.tensor([[4]]):
                performed_actions = 0
            return action
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        #print('random:', action)
        if action == torch.tensor([[4]]):
            performed_actions = 0
        return action


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train:# and random.random() < random_prob:
        action = select_action(self, game_state)
        return ACTIONS[action[0][0]]

    self.logger.debug("Querying model for action.")
    #print(game_state)
    action = select_action(self, game_state)
    return ACTIONS[action[0][0]]


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
    
    # Q MATRICES
    #Q = np.full((17, 17), 4)
    #if len(game_state['coins']) != 0:
    #    coin = game_state['coins'][0]
    #    Q = Q_matrices[coin[1], coin[0]]
    #    Q = Q.argmax(axis=2)
        
    # For example, you could construct several channels of equal shape, ...

    # EXPLOSION MAP
    explosion_map = np.array(game_state['explosion_map'])
    explosion_map = explosion_map.transpose()

    position = game_state['self'][3]
    #arena[position[1], position[0]] = 10  # -1=block, 0=free, 1=crate
    #Q[position[1], position[0]] = 10
    #explosion_map[position[1], position[0]] = 10

    # POSITIONS 
    positions = np.full((17,17), 0)
    positions[position[1], position[0]] = 1
    
    positions = np.full((17,17), 0)
    positions[position[1], position[0]] = 1
    for other in game_state['others']:
        positions[other[3][1], other[3][0]] = -1

    # ARENA
    arena = np.array(game_state['field'])
    arena = arena.transpose()

    #arena[position[1], position[0]] = 0.5
    #if len(game_state['bombs']) != 0:
    #    for bomb in game_state['bombs']:
    #        bomb_x = bomb[0][0]
    #        bomb_y = bomb[0][1]
    #        arena[bomb_y, bomb_x] = -0.5
    
    # COINS & BOMBS
    coins_matrix = np.full((17, 17), 0)
    if len(game_state['coins']) != 0:
        for coin in game_state['coins']:
            coins_matrix[coin[1], coin[0]] = 1

    bomb_matrix = np.full((17, 17), 0)
    if len(game_state['bombs']) != 0:
        for bomb in game_state['bombs']:
            bomb_x = bomb[0][0]
            bomb_y = bomb[0][1]
            for i in range(4):
                if bomb_y-i > 0 and arena[bomb_y-i, bomb_x] != -1:
                    bomb_matrix[bomb_y-i, bomb_x] = -1
                else:
                    break
            for i in range(4):
                if bomb_y+i < 17 and arena[bomb_y+i, bomb_x] != -1:
                    bomb_matrix[bomb_y+i, bomb_x] = -1
                else:
                    break
            for i in range(4):
                if bomb_x-i > 0 and arena[bomb_y, bomb_x-i] != -1:
                    bomb_matrix[bomb_y, bomb_x-i] = -1
                else:
                    break
            for i in range(4):
                if bomb_x+i < 17 and arena[bomb_y, bomb_x+i] != -1:
                    bomb_matrix[bomb_y, bomb_x+i] = -1
                else:
                    break

    bomb_summary = -np.logical_or(bomb_matrix, explosion_map).astype(int)
    if len(game_state['coins']) != 0:
        for coin in game_state['coins']:
            bomb_matrix[coin[1], coin[0]] = 1
            bomb_summary[coin[1], coin[0]] = 1

    channels = list()
    channels.append(arena)
    channels.append(bomb_summary) #!
    #channels.append(explosion_map)
    #channels.append(Q)
    channels.append(positions)
    #channels.append(coins_matrix)
    #channels.append(bomb_matrix)
    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    #stacked_channels.reshape(-1)

    tensor_channels = torch.FloatTensor(channels)
    
    tensor_channels = tensor_channels.unsqueeze(0)
    return tensor_channels
