import pickle
import random
from collections import namedtuple, deque, defaultdict
from typing import List
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import events as e
from .callbacks import state_to_features, select_action

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

BATCH_SIZE = 64 # originally 128
GAMMA = 0.7
TARGET_UPDATE = 10

n_actions = 6
steps_done = 0

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 2, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    #self.global_policy_net = DQN(17, 17, n_actions).to(device)
    #self.global_target_net = DQN(17, 17, n_actions).to(device)
    #self.global_target_net.load_state_dict(self.global_policy_net.state_dict())
    #self.global_target_net.eval()

    self.transitions = ReplayMemory(10000)
    self.optimizer = optim.RMSprop(self.global_policy_net.parameters())

    self.i_episode = 0

losses = []

def optimize_model(self):
    if len(self.transitions) < BATCH_SIZE:
        return
    transitions = self.transitions.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = self.global_policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = self.global_target_net(non_final_next_states).max(1)[0].detach()
    
    # Calculate expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    losses.append(loss)
    with open('loss.txt', 'w') as doc:
        for l in losses:
            doc.write(str(l.item()) + '\n')

    self.optimizer.zero_grad()
    loss.backward()
    for param in self.global_policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

positions = list()


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

    actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']

    global positions

    if old_game_state == None or new_game_state == None or select_action == None:
        return

    if new_game_state['self'][3] in positions:
        events.append('POSITION_REPEATED')

    if len(positions) == 3:
        positions.pop(0)
    positions.append(new_game_state['self'][3])

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    self.transitions.push(state_to_features(old_game_state), torch.tensor([actions.index(self_action)]), state_to_features(new_game_state), torch.tensor([reward_from_events(self, events)]))

    optimize_model(self)
    

rewards = []
all_rewards = []

round_events = defaultdict(lambda: 0)
all_events = []

ev = [
        e.MOVED_UP,
        e.MOVED_DOWN,
        e.MOVED_LEFT,
        e.MOVED_RIGHT,
        e.COIN_COLLECTED,
        e.KILLED_SELF,
        e.BOMB_DROPPED,
        e.SURVIVED_ROUND,
        e.INVALID_ACTION,
        e.WAITED,
        'SURVIVED_BOMB',
        'STUPID',
        'POSITION_REPEATED',
        'SURVIVED_MOVE'
]

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'BOMB', 'WAIT']

    global positions
    positions = list()

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.push(state_to_features(last_game_state), torch.tensor([actions.index(last_action)]), state_to_features(last_game_state), torch.tensor([reward_from_events(self, events)]))    

    # Update the target network, copying all weights and biases in DQN
    if self.i_episode % TARGET_UPDATE == 0:
        self.global_target_net.load_state_dict(self.global_policy_net.state_dict())

    self.i_episode += 1

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        self.model = (self.global_policy_net, self.global_target_net)
        pickle.dump(self.model, file)

    global rewards

    with open('rewards.txt', 'w') as doc:
        for reward in all_rewards:
            doc.write(str(sum(reward)) + '\n')
    all_rewards.append(rewards)
    rewards = []

    global all_events
    global round_events
    all_events.append(round_events)
    round_events = defaultdict(lambda: 0)

    with open('events.txt', 'w') as doc:
        for r in all_events:
            elem = [str(r[k]) for k in ev]
            line = ';'.join(elem) + '\n'
            doc.write(line)




def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    
    if e.BOMB_EXPLODED in events and e.KILLED_SELF not in events and e.CRATE_DESTROYED in events:
        events.append('SURVIVED_BOMB')
    if e.BOMB_EXPLODED in events and e.KILLED_SELF not in events and e.CRATE_DESTROYED not in events:
        events.append('STUPID')

    if e.GOT_KILLED not in events:
        events.append('SURVIVED_MOVE')

    for event in events:
        round_events[event] += 1

    game_rewards = {
        e.MOVED_UP: 0.2,
        e.MOVED_DOWN: 0.2,
        e.MOVED_LEFT: 0.2,
        e.MOVED_RIGHT: 0.2,
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 1,
        #e.COIN_FOUND: 0.1,
        #e.KILLED_SELF: -1,
        e.BOMB_DROPPED: 0.3,
        e.SURVIVED_ROUND: 1,
        #e.INVALID_ACTION: -1,
        #e.WAITED: -0.2,
        'SURVIVED_BOMB': 1,
        'STUPID': 0.5,
        'POSITION_REPEATED': -0.2,
        'SURVIVED_MOVE': 0.2,
        e.GOT_KILLED: -1
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    rewards.append(reward_sum)
    return reward_sum
