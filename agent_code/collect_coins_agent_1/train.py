import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features
from .callbacks import get_next_action, get_next_location

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
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
    self.coins_per_epoch = list()
    self.reward_per_epoch = list()
    self.invalid_actions_per_epoch = list()
    self.coins = 0
    self.total_reward = 0
    self.total_invalid_actions = 0
    

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)
    ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    
    self.coin_x, self.coin_y = new_game_state['coins'][0]

    if old_game_state == None:
        return
    old_row_index, old_column_index = old_game_state['self'][3]
    row_index, column_index = new_game_state['self'][3]
        
    discount_factor = 0.8
    learning_rate = 0.9

    reward = 0

    if e.INVALID_ACTION in events:
        self.total_invalid_actions += 1

    if (old_column_index == column_index) and (old_row_index == row_index) and self_action != 'WAIT': #Maybe rather check if "invalid move" is in events
        action_index = ACTIONS.index(self_action)
        invalid_position = get_next_location(column_index, row_index, action_index)
        reward = -100
        old_q_value = self.Q[invalid_position[1], invalid_position[0], action_index]
        temporal_difference = reward + (discount_factor * np.max(self.Q[column_index, row_index])) - old_q_value
    
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        self.Q[old_column_index, old_row_index, action_index] = new_q_value

    elif len(new_game_state['coins']) > 0 and self_action != None:
        action_index = ACTIONS.index(self_action)
        reward = self.R[column_index, row_index]
        old_q_value = self.Q[old_column_index, old_row_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(self.Q[column_index, row_index])) - old_q_value
    
        new_q_value = old_q_value + (learning_rate * temporal_difference)
        self.Q[old_column_index, old_row_index, action_index] = new_q_value
    if e.COIN_COLLECTED in events:
        self.R[column_index, row_index] = -1.
        self.coins += 1
        
    self.total_reward += reward

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.coins_per_epoch.append(self.coins)
    self.reward_per_epoch.append(self.total_reward)
    self.invalid_actions_per_epoch.append(self.total_invalid_actions)
    print(self.coins, 'coins collected')
    print(self.total_reward, 'total reward')
    print(self.total_invalid_actions, 'total invalid actions')
    self.logger.debug('coins collected: ' + str(self.coins))
    self.logger.debug('total reward: ' + str(self.total_reward))
    self.logger.debug('total invalid actions: '+ str(self.total_invalid_actions))

    self.R = np.full((17, 17), -1.)
    self.R[0,:] = -100.
    self.R[-1:,:] = -100.
    self.R[:,0] = -100.
    self.R[:, -1] = -100.
    for row in range(2,15):
        for column in range(2, 15):
            if (column % 2 == 0) and (row % 2 == 0):
                self.R[row, column] = -100.
    self.Q_matrices[self.coin_y, self.coin_x] = self.Q.copy()
    with open('statistics.txt', 'w') as doc:
        for i in range(len(self.coins_per_epoch)):
            doc.write(str(self.coins_per_epoch[i]) + '\t' + str(int(self.reward_per_epoch[i])) + '\t' + str(self.invalid_actions_per_epoch[i]) + '\n')

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        self.model = self.Q_matrices
        pickle.dump(self.model, file)

    self.coins = 0
    self.total_reward = 0
    self.total_invalid_actions = 0
    self.i = 0
    print('c', self.coin_y, self.coin_x)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
