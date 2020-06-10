import numpy as np
import deep_network
class actor_critic:

    def __init__(self, actor_shape, critic_shape, loss_func, learning_rate, reg):
        self.alpha = learning_rate
        self.reg = reg
        self.actor_shape = actor_shape
        self.critic_shape = critic_shape
        self.actor = deep_network.deep_network(actor_shape, sigmoid, d_sigmoid)
        self.critic = deep_network.deep_network(critic_shape, sigmoid, d_sigmoid)
        self.states = []
        self.a_s = []
        self.z_s = []
        self.loss_func = loss_func

    def act(self, state):
        action, a, z = self.actor.feed_forward(state)
        self.states.append(state.copy())
        self.a_s.append(a)
        self.z_s.append(z)
        return action

    def learn(self):
        state_shape = [len(self.states), self.states[0].shape[0]]
        states = np.concatenate(self.states).reshape(state_shape)
        #print(np.round(states,3))
        layers = len(self.a_s[0])
        a_s = []
        z_s = []
        for l in range(layers):
            a_cols = [row[l] for row in self.a_s]
            a_s.append(np.array(a_cols))

        for l in range(layers-1):
            z_cols = [row[l] for row in self.z_s]
            z_s.append(np.array(z_cols))

        #(layers)
        experimental_punishment = self.loss_func(states)
        print("Avg loss:", np.mean(experimental_punishment))
        critic_input = np.concatenate((a_s[-1], states), axis = 1)
        critic_guesses, c_a, c_z = self.critic.feed_forward(critic_input)
        d_output_critic = np.subtract(critic_guesses , experimental_punishment[..., np.newaxis])
        print("Critic loss:", np.mean(np.abs(d_output_critic)))
        self.critic.backpropagate(d_output_critic, c_a, c_z, self.alpha, self.reg)
        actor_output_size = self.actor.shape[-1]
        d_output_actor = self.critic.backpropagate(d_output_critic, c_a, c_z, 0, self.reg)[:,:actor_output_size]
        self.actor.backpropagate(d_output_actor, a_s, z_s, self.alpha, self.reg)

        self.clear_stored()

    def clear_stored(self):
        self.states = []
        self.a_s = []
        self.z_s = []


def swish(z):
    return z / (1 + np.exp(-z))
def d_swish(z):
    sg = 1 / (1 + np.exp(-z))
    sw = z * sg
    return sw + sg * (1 - sw)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def d_sigmoid(z):
    sig = sigmoid(z)
    return np.multiply(sig,(1-sig))