import gym
import numpy as np
from matplotlib import pyplot as plt
from dqn_agent import DQNAgent
from ddpg import DDPG
import torch
import tqdm
import time
import hydra
from pathlib import Path
import wandb
import warnings
import make_env
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from common import helper as h
from common import logger as logger
from common.buffer import ReplayBuffer


@hydra.main(config_path='cfg', config_name='project_cfg')
def main(cfg):
    # set random seed
    h.set_seed(cfg.seed)

    # create folders if needed
    work_dir = Path().cwd()/cfg.env_name
    model_path = work_dir 
    # create env
    if cfg.env_name =="lunarlander":
        if cfg.agent_name == "dqn":
            env=make_env.create_env(config_file_name="lunarlander_discrete_easy", seed=cfg.seed)
            n_actions = env.action_space.n
        if cfg.agent_name == "ddpg":
            env=make_env.create_env(config_file_name="lunarlander_continuous_easy", seed=cfg.seed)
            action_dim = env.action_space.shape[0]
            max_action = env.action_space.high[0]
    elif cfg.env_name =="mountaincar":
        if cfg.agent_name == "dqn":
            env=make_env.create_env(config_file_name="mountaincarcontinuous_easy", seed=cfg.seed)
        if cfg.agent_name == "ddpg":
            env=make_env.create_env(config_file_name="mountaincarcontinuous_easy", seed=cfg.seed)
            action_dim = env.action_space.shape[0]
            max_action = env.action_space.high[0]
    if cfg.save_video:
        env = gym.wrappers.RecordVideo(env, work_dir/'video'/'test', 
                                        episode_trigger=lambda x: True,
                                        name_prefix=cfg.exp_name)
    # get number of actions and state dimensions
    
    state_shape = env.observation_space.shape

    # init agent
    if cfg.agent_name == "dqn":
        agent = DQNAgent(state_shape, n_actions, batch_size=cfg.batch_size, hidden_dims=cfg.hidden_dims,
                         gamma=cfg.gamma, lr=cfg.lr, tau=cfg.tau)
    elif cfg.agent_name == "ddpg":
        agent = DDPG(state_shape, action_dim, max_action, cfg.lr, cfg.gamma, cfg.tau, cfg.batch_size, cfg.buffer_size)
    else:
        raise ValueError(f"No {cfg.agent_name} agent implemented")

    # Load policy / q functions
    agent.load(model_path)

    for ep in range(cfg.test_episodes):
        state, done, ep_reward, env_step = env.reset(), False, 0, 0
        rewards = []

        # collecting data and fed into replay buffer
        while not done:
            # Select and perform an action
            action = agent.get_action(state, epsilon=0.0)
            if isinstance(action, np.ndarray): action = action.item()
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            rewards.append(reward)

        info = {'episode': ep, 'ep_reward': ep_reward}
        if (not cfg.silent): print(info)


if __name__ == '__main__':
    main()