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
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from common import helper as h
from common import logger as logger
from common.buffer import ReplayBuffer

import make_env

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

def dis_to_con(discrete_action, env, action_dim):
    action_low_bound = env.action_space.low
    action_up_bound = env.action_space.high
    action = action_low_bound + (discrete_action / (action_dim - 1)) * (action_up_bound - action_low_bound)

    return action

def train(agent, env, max_episode_steps=1000):
    # Run actual training        
    reward_sum, timesteps, done, episode_timesteps = 0, 0, False, 0
    # Reset the environment and observe the initial state
    obs = env.reset()
    while not done:
        episode_timesteps += 1
        
        # Sample action from policy
        action, act_logprob = agent.get_action(obs)

        # Perform the action on the environment, get new state and reward
        next_obs, reward, done, _ = env.step(to_numpy(action))

        # Store action's outcome (so that the agent can improve its policy)
        if isinstance(agent, DDPG):
            # ignore the time truncated terminal signal
            done_bool = float(done) if episode_timesteps < max_episode_steps else 0 
            agent.record(obs, action, next_obs, reward, done_bool)
        else: raise ValueError

        # Store total episode reward
        reward_sum += reward
        timesteps += 1

        # update observation
        obs = next_obs.copy()

    # update the policy after one episode
    info = agent.update()

    # Return stats of training
    info.update({'timesteps': timesteps,
                'ep_reward': reward_sum,})
    return info


@hydra.main(config_path='cfg', config_name='project_cfg')
def main(cfg):
    # set random seed
    h.set_seed(cfg.seed)

    cfg.run_id = int(time.time())
    # create folders if needed
    work_dir = Path().cwd()/cfg.env_name
    if cfg.save_model:
        model_path = work_dir 
        h.make_dir(model_path)
    
    # use wandb to store stats
    if cfg.use_wandb:
        wandb.init(project="rl_aalto",
                    name=f'{cfg.exp_name}-{cfg.agent_name}-{cfg.env_name}-{str(cfg.seed)}-{cfg.run_id}',
                    group=f'{cfg.exp_name}-{cfg.env_name}',
                    config=cfg)
    
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
            n_actions = 11 # to do set manually
        if cfg.agent_name == "ddpg":
            env=make_env.create_env(config_file_name="mountaincarcontinuous_easy", seed=cfg.seed)
            action_dim = env.action_space.shape[0]
            max_action = env.action_space.high[0]
    
   
    # env.seed(cfg.seed)
    if cfg.save_video:
        env = gym.wrappers.RecordVideo(env, work_dir/'video'/'train',
                                        episode_trigger=lambda x: x % 100 == 0,
                                        name_prefix=cfg.exp_name) # save video for every 100 episodes
    # get number of actions and state dimensions
    
   
   
   
    # 
    state_shape = env.observation_space.shape


    # init agent
    if cfg.agent_name == "dqn":
        agent = DQNAgent(state_shape, n_actions, batch_size=cfg.batch_size, hidden_dims=cfg.hidden_dims,
                         gamma=cfg.gamma, lr=cfg.lr, tau=cfg.tau)
    elif cfg.agent_name == "ddpg":
        # agent = DDPG(state_shape, action_dim, max_action, cfg.lr, cfg.gamma, cfg.tau, cfg.batch_size, cfg.buffer_size)
        agent = DDPG(state_shape, action_dim, max_action, cfg.actor_lr, cfg.gamma, cfg.tau, cfg.batch_size, cfg.buffer_size)
        # pass
    else:
        raise ValueError(f"No {cfg.agent_name} agent implemented")

   
    
    
    if cfg.agent_name == "dqn":
        buffer = ReplayBuffer(state_shape, action_dim=1, max_size=int(cfg.buffer_size))

        #start training
        time_start=time.time()
        for ep in range(cfg.train_episodes):
            state, done, ep_reward, env_step = env.reset(), False, 0, 0
            eps = max(cfg.glie_b/(cfg.glie_b + ep), 0.05)

            # collecting data and fed into replay buffer
            while not done:
                env_step += 1
                if ep < cfg.random_episodes: # in the first #random_episodes, collect random trajectories
                    if cfg.env_name == "mountaincar":
                        num_action = agent.n_actions
                        action = np.random.randint(low=0, high=num_action)
                    else:
                        action = env.action_space.sample()
                else:
                    # Select and perform an action
                    action = agent.get_action(state, eps)
                    if isinstance(action, np.ndarray): action = action.item()
                
                if cfg.env_name == "mountaincar":
                    action = dis_to_con(action, env, agent.n_actions)
                    
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward

                # Store the transition in replay buffer
                buffer.add(state, action, next_state, reward, done)

                # Move to the next state
                state = next_state
            
                # Perform one update_per_episode step of the optimization
                if ep >= cfg.random_episodes:
                    update_info = agent.update(buffer)
                else: update_info = {}

            info = {'episode': ep, 'epsilon': eps, 'ep_reward': ep_reward}
            info.update(update_info)

            if cfg.use_wandb: wandb.log(info)
            if (not cfg.silent) and (ep % 100 == 0): print(info)
                
    elif cfg.agent_name == "ddpg":
        
        time_start=time.time()
        for ep in range(cfg.train_episodes + 1):
            # collect data and update the policy
            train_info = train(agent, env)

            if cfg.use_wandb:
                wandb.log(train_info)
            if (not cfg.silent) and (ep % 100 == 0):
                print({"ep": ep, **train_info})

        
    if cfg.save_model:
        agent.save(model_path)
    time_end=time.time()
    
    print('------ Training Finished ------')
    print('time cost',time_end-time_start,'s')


if __name__ == '__main__':
    main()