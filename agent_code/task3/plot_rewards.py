import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import math

coins = list()
total_reward = list()
total_invalid_moves = list()

with open('rewards.txt', 'r') as doc:
    for line in doc.readlines():
        line = line.strip('\n')
        #line = line.split('\t')
        #print(line)
        line = float(line)
        #coins.append(line[0])
        total_reward.append(line)
        #total_invalid_moves.append(line[2])

fig, ax = plt.subplots(1,1, figsize=(12,9))
#ax[0].set_title('coins per round')
#ax[0].plot(coins)
ax.set_title('total reward per round')
ax.plot(total_reward)
#ax[2].set_title('total invalid moves per round')
#ax[2].plot(total_invalid_moves)
fig.tight_layout()
fig.savefig('statistics.png')

plt.close()

loss = list()

with open('loss.txt', 'r') as doc:
    for line in doc.readlines():
        line = line.strip('\n')
        #line = line.split('\t')
        #print(line)
        line = float(line)
        #coins.append(line[0])
        loss.append(line)
        #total_invalid_moves.append(line[2])

ev = [
        'MOVED_UP',
        'MOVED_DOWN',
        'MOVED_LEFT',
        'MOVED_RIGHT',
        'COIN_COLLECTED',
        #e.KILLED_OPPONENT: 0.4,
        #e.COIN_FOUND: 0.1,
        'KILLED_SELF',
        'BOMB_DROPPED',
        'SURVIVED_ROUND',
        'INVALID_ACTION',
        'WAITED',
        'SURVIVED_BOMB',
        'STUPID',
        'POSITION_REPEATED',
        'SURVIVED_MOVE'
]

to_plot = ['SURVIVED_BOMB']

events = dict()
for e in ev:
    events[e] = list()

with open('events.txt', 'r') as doc:
    mylist = doc.read().splitlines()
    for line in mylist:
        line = line.split(';')
        for i, elem in enumerate(line):
            events[ev[i]].append(int(elem))

fig, ax = plt.subplots(1,1, figsize=(12,9))
ax.set_title('events per round')
for e in ev:
    if e in to_plot:
        print(len(events[e]))
        x = np.arange(0, len(events[e]))
        y = events[e]
        X_Y_Spline = make_interp_spline(x, y)
        X_ = np.linspace(x.min(), x.max(), 500)
        Y_ = X_Y_Spline(X_)
        #ax.plot(events[e], label=e)
        ax.plot(X_, Y_, label=e)
plt.legend(loc="upper left")

fig.tight_layout()
fig.savefig('events.png')
plt.close()

fig, ax = plt.subplots(1,1, figsize=(12,9))
ax.set_title('events per round')
ax.plot(np.array(events['SURVIVED_BOMB']) + np.array(events['STUPID']), label=e)
plt.legend(loc="upper left")

fig.tight_layout()
fig.savefig('survived_bombs_total.png')
plt.close()



fig, ax = plt.subplots(1,1, figsize=(12,9))
#ax[0].set_title('coins per round')
#ax[0].plot(coins)
ax.set_title('loss per iteration')
ax.set_yscale('log')
x = np.arange(0, len(loss))
y = loss
X_Y_Spline = make_interp_spline(x, y)
X_ = np.linspace(x.min(), x.max(), 1000)
Y_ = X_Y_Spline(X_)
#ax.plot(X_, Y_)
ax.plot(loss)
#ax[2].set_title('total invalid moves per round')
#ax[2].plot(total_invalid_moves)
fig.tight_layout()
fig.savefig('loss.png')

plt.close()

fig, ax = plt.subplots(1, 1, figsize=(12,9))
ax.set_title('Epsilon per round')

steps_done = np.arange(1, 10000)
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 5000
eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
ax.plot(steps_done, eps_threshold)
fig.tight_layout
fig.savefig('epsilon.png')
