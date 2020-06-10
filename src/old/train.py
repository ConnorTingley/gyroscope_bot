from actor_critic_tf import Agent
from utils import plotLearning
import numpy as np

if __name__ == '__main__':
    agent = Agent(alpha = 1e-5, beta = 5e-5)
    score_history = []
    num_episodes = 2000

    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward,done,info = env.step(action)
            agent.learn(observation,action,reward,observation_,done)
            observation = observation_
            score += reward

        score_history.append(score)
        avg_score = np.mean(score_history[-100:0])
        print('episode ', i , "score %.2f average score %.2f" % (score,avg_score))

        filename = 'gyroscope-bot-actor-critic.png'
        plotLearning(score_history, filename=filename, window=100)