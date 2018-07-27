import gym.spaces
from zuofengRL.deepq.brain import Brain
from zuofengRL.deepq.policy import MLP
import visdom
import matplotlib.pyplot as plt
import numpy as np


def nature_dqn():
    env = gym.make("CartPole-v0")
    env.observation_space
    agent = Brain(env=env, model=MLP, batch_size=32, lr=0.00025, max_timestep=50000,
                  buffer_size=50000, learning_start=1000, target_network_update_freq=500, env_name="cartpole",
                  double_q=False, dueling=False, prioritized=False)
    mr = agent.run_loop()
    return mr


def double_dqn():
    env = gym.make("CartPole-v0")
    agent = Brain(env=env, model=MLP, batch_size=32, lr=0.00025, max_timestep=50000,
                  buffer_size=50000, learning_start=1000, target_network_update_freq=500, env_name="cartpole",
                  double_q=True, dueling=False, prioritized=False)
    mr = agent.run_loop()
    return mr


def dueling_dqn():
    env = gym.make("CartPole-v0")
    agent = Brain(env=env, model=MLP, batch_size=32, lr=0.00025, max_timestep=50000,
                  buffer_size=50000, learning_start=1000, target_network_update_freq=500, env_name="cartpole",
                  double_q=False, dueling=True, prioritized=False)
    mr = agent.run_loop()
    return mr


def prioritized_dqn():
    env = gym.make("CartPole-v0")
    agent = Brain(env=env, model=MLP, batch_size=32, lr=0.00025, max_timestep=50000,
                  buffer_size=50000, learning_start=1000, target_network_update_freq=500, env_name="cartpole",
                  double_q=False, dueling=False, prioritized=True)
    mr = agent.run_loop()
    return mr


def all_dqn():
    env = gym.make("CartPole-v0")
    agent = Brain(env=env, model=MLP, batch_size=32, lr=0.00025, max_timestep=50000,
                  buffer_size=50000, learning_start=1000, target_network_update_freq=500, env_name="cartpole",
                  double_q=True, dueling=True, prioritized=True, use_cuda=False)
    mr = agent.run_loop()
    return mr


if __name__ == '__main__':
    nature_mr = nature_dqn()
    double_mr = double_dqn()
    dueling_mr = dueling_dqn()
    prior_mr = prioritized_dqn()
    all_mr = all_dqn()
    # min_len = np.min([len(nature_mr), len(double_mr), len(dueling_mr), len(prior_mr), len(all_mr)])
    #
    # nature_mr = nature_mr[:min_len]
    # double_mr = double_mr[:min_len]
    # dueling_mr = dueling_mr[:min_len]
    # prior_mr = prior_mr[:min_len]
    # all_mr = all_mr[:min_len]
    
    vis = visdom.Visdom()
    ax = plt.figure(1)
    plt.plot(nature_mr, label="nature dqn r")
    plt.plot(double_mr, label="double dqn r")
    plt.plot(dueling_mr, label="dueling dqn r")
    plt.plot(prior_mr, label="prior dqn r")
    plt.plot(all_mr, label="all dqn r")
    plt.xlabel("steps")
    plt.ylabel("mean_reward")
    plt.legend(loc="best")
    vis.matplot(ax)











