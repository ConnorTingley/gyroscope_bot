import numpy as np
import actor_critic

def state_init():
    state = np.zeros(2)
    state[0] = 0.75
    state[1] = 0.25
    return state

def state_update(state,player_move):
    move_rate = 0.1
    move = move_rate if np.abs((state[0] % 0.1) - 0.05) < 0.001 else -move_rate
    state[0] += move
    if (state[0] < 0):
        state[0] = 0.05
    if (state[0] > 1):
        state[0] = 1
    state[1] = player_move
    return state

def loss(states):

    return np.abs(states[:,0] - states[:,1])


def keep_this_around():
    i = 0
    # must change
    #posy = states[:, 1]
    #total_momentum = states[:, 4] + states[:, 5] + states[:, 6]
    #death = 0 if posy > 0 else 1  # needs to be the sum over the next second

if __name__ == "__main__":
    ac = actor_critic.actor_critic([2, 50, 20, 1], [3, 100, 50, 1], loss, 0.0005, 0)
    for e in range(20000):
        state = state_init()
        for i in range(500):
            move = ac.act(state)
            state = state_update(state,move)
        print(move)
        ac.learn()
    print(ac.critic.feed_forward(np.array([0.9, 1, 0])))
    print(ac.critic.feed_forward(np.array([0, 0, 0])))

