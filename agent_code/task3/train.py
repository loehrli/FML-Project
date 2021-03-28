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
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

BATCH_SIZE = 64 # originally 128
GAMMA = 0.7
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
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
        #self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        #self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        #self.drop = nn.Dropout(0.1666)
        #self.flat = nn.Flatten()
        #self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input 289image size, so compute it.
        def conv2d_size_out(size, kernel_size = 2, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
        #self.head = nn.Linear(256, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
        #x = F.relu(self.conv(x))
        #x = self.pool(x)
        #x = x.view(x.size(0), -1)
        #x = self.drop(x)
        #x = self.head(x)
        #return x

#def select_action(self, state):
#    
#    #global steps_done
#    sample = random.random()
#    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#        math.exp(-1. * steps_done / EPS_DECAY)
#    steps_done += 1
#    if sample > eps_threshold:
#        with torch.no_grad():
#            # t.max(1) will return largest column value of each row.
#            # second column on max result is index of where max element was
#            # found, so we pick action with the larger expected reward.
#            return self.global_policy_net(state).max(1)[1].view(1, 1)
#    else:
#        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


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
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    #print(len(transitions))

    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = self.global_policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = self.global_target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    losses.append(loss)
    with open('loss.txt', 'w') as doc:
        for l in losses:
            doc.write(str(l.item()) + '\n')

    # Optimize the model
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

    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    # Store the transition in memory
    self.transitions.push(state_to_features(old_game_state), torch.tensor([actions.index(self_action)]), state_to_features(new_game_state), torch.tensor([reward_from_events(self, events)]))

    # Observe new state
    #last_screen = current_screen
    #current_screen = get_screen()
    #if not done:
    #    next_state = current_screen - last_screen
    #else:
    #    next_state = None

    # Move to the next state
    #state = next_state

    # Perform one step of the optimization (on the target network)
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
        #e.KILLED_OPPONENT: 0.4,
        #e.COIN_FOUND: 0.1,
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


    #game_rewards = {
    #    e.MOVED_UP: 0.2,
    #    e.MOVED_DOWN: 0.2,
    #    e.MOVED_LEFT: 0.2,
    #    e.MOVED_RIGHT: 0.2,
    #    e.COIN_COLLECTED: 1,
    #    #e.KILLED_OPPONENT: 0.4,
    #    #e.COIN_FOUND: 0.1,
    #    e.KILLED_SELF: -1,
    #    e.BOMB_DROPPED: 1,
    #    e.SURVIVED_ROUND: 1,
    #    e.INVALID_ACTION: -1,
    #    e.WAITED: -0.2,
    #    'SURVIVED_BOMB': 1,
    #    'STUPID': 0.5,
    #    'POSITION_REPEATED': -0.2,
    #    'SURVIVED_MOVE': 0.2
    #}
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
        e.GOT_KILLED: -1,
        #e.OPPONENT_ELIMINATED: 1
    } #next: gamma higher again & comment out wait, invalid action, killed self, stupid
    # gamma 0.4, rewards above try more rounds

    #ame_rewards = {
    #    'SURVIVED_BOMB': 0.5,
    #    'COIN_COLLECTED': 1
    #    #'SURVIVED_MOVE': 0.2,
    #    #'POSITION_REPEATED': -0.2
    #}

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    rewards.append(reward_sum)
    print(events)
    #self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
