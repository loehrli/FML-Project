import matplotlib.pyplot as plt

coins = list()
total_reward = list()
total_invalid_moves = list()

with open('statistics.txt', 'r') as doc:
    for line in doc.readlines():
        line = line.strip('\n')
        line = line.split('\t')
        print(line)
        line = [int(elem) for elem in line]
        coins.append(line[0])
        total_reward.append(line[1])
        total_invalid_moves.append(line[2])

fig, ax = plt.subplots(3,1, figsize=(12,9))
ax[0].set_title('coins per round')
ax[0].plot(coins)
ax[1].set_title('total reward per round')
ax[1].plot(total_reward)
ax[2].set_title('total invalid moves per round')
ax[2].plot(total_invalid_moves)
fig.tight_layout()
fig.savefig('statistics.png')
