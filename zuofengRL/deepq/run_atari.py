import gym.spaces
from zuofengRL.deepq.brain import Brain
from zuofengRL.deepq.policy import MLP, CNN
from zuofengRL.deepq.atari_wrappers import make_atari,wrap_deepmind


def nature_dqn():
    env = make_atari("BreakoutNoFrameskip-v4")
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    agent = Brain(env=env, model=CNN, batch_size=32, lr=0.00025, max_timestep=10000000,
                  buffer_size=100000, learning_start=50000, target_network_update_freq=10000, env_name="breakout",
                  double_q=True, dueling=True, prioritized=True, atari=True, use_cuda=False)
    mr = agent.run_loop()
    return mr


if __name__ == '__main__':
    nature_dqn()













