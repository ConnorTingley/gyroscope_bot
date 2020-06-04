import numpy as np
import deep_network
class actor_critic:

    def __init__(self, actor_shape, critic_shape, alpha, momentum_punish, overshoot_punish, death_punish):
        if actor_shape[-1] != critic_shape[0]:
            print("A-C shapes missaligned: {actor_shape[-1]} != {critic_shape[0]}")
        self.alpha = alpha
        self.actor_shape = actor_shape
        self.critic_shape = critic_shape
        self.actor = deep_network(actor_shape)
        self.critic = deep_network(critic_shape)
        self.states = []
        self.a_s = []
        self.z_s = []

        self.momentum_punish = momentum_punish
        self.overshoot_punish = overshoot_punish
        self.death_punish = death_punish

    def act(self, state):
        action = self.actor.feed_forward(self, state)
        self.states.append(state)
        self.a_s.append(self.actor.a)
        self.z_s.append(self.actor.z)
        return action

    def learn(self, overshoots):
        for i in range(len(self.states)):
            state = self.state[i]
            a = self.a_s[i]
            z = self.z_s[i]
            critic_input = np.concatenate(state,a[-1])
            # must change
            posy = state[1]
            total_moment = state[4] + state[5] + state[6]
            death = 0 if posy > 0 else 1 # needs to be the sum over the next second

            experimental_punishment = np.square(posy) + np.square(self.momentum_punish * total_moment) + self.death_punish * death + self.overshoot_punish * overshoots[i]
            critic_guess = self.critic.feed_forward(critic_input)
            d_output_critic = critic_guess - experimental_punishment
            self.critic.backpropogate(d_output_critic, self.alpha)

            self.critic.backpropagate_no_update(experimental_punishment)
            d_output_actor = np.multiply(self.critic.W[0].T @ self.critic.d[0], deep_network.dsigmoid(self.actor.z[-1]))
            self.actor.a = a
            self.actor.z = z
            self.actor.backpropogate(d_output_actor, self.alpha)

        self.critic.update()
        self.actor.update()

        self.clear_stored()

    def clear_stored(self):
        self.states = []
        self.a_s = []
        self.z_s = []