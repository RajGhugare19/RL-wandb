import numpy as np 
import random 
import gym 
import gym_minigrid
from gym_minigrid.wrappers import *
from wrappers import MDPWrapper

import torch 
import torch.nn as nn 
import torch.optim as optim 

from utils import DQNMemory
from models import DQNModel

device = torch.device("cuda:0")

import wandb
wandb.login()

config = dict(
    learning_rate=0.003,
    batch_size=512,
    total_iterations=20000,
    target_update=500,
    gamma=0.99,
    memory_size=20000,
    log_step=200,
    env = 'MiniGrid-Empty-5x5-v0',
    state_size = 36,
    action_size = 3,
    architecture = "DQN",
    render = "True"
)

def make(config):
    
    q_model = DQNModel(config).to(device)
    q_target_model = DQNModel(config).to(device)
    q_target_model.load_state_dict(q_model.state_dict())
    replay_buffer = DQNMemory(config)
    return q_model, q_target_model, replay_buffer


def train(q_model,q_target_model,replay_buffer,config):
    
    epsilon = 1
    dqn_gamma = config['gamma']
    batch_size = config['batch_size']
    batch_index = [i for i in range(batch_size)]
    target_update = config['target_update']
    log_step = config['log_step']
    total_iterations = config['total_iterations']
    eps_decay = (1-0.05)/total_iterations
    
    env = gym.make(config['env'])
    env = FullyObsWrapper(env)
    env = MDPWrapper(env)
    n_states = np.product(env.observation_space.shape)
    done = False
    s = env.reset()
    #s = state['image'].reshape(n_states)
    score = 0
    
    for n in range(1,total_iterations):

        a = q_model.action(s,epsilon)
        ns,r,done,_ = env.step(a)
        #ns = next_state['image'].reshape(n_states)
        replay_buffer.store(s,a,r,done,ns)
        score += r
        
        if done:
            done = False 
            s = env.reset()
            #s = state['image'].reshape(n_states)
            wandb.log({"reward per episode": score}, step=n)
            print(f"score after " + str(n) + f" frames: {score:.3f}")
            score = 0
        else:    
            s = ns

        if replay_buffer.memory_count > batch_size:

            s_batch,a_batch,r_batch,term_batch,ns_batch = replay_buffer.sample()
            
            s_batch = torch.tensor(s_batch).to(device)
            ns_batch = torch.tensor(ns_batch).to(device)
            r_batch = torch.tensor(r_batch).to(device)
            term_batch = torch.tensor(term_batch).to(device)
            a_batch = torch.tensor(a_batch).to(device)

            q_val = q_model(s_batch)[batch_index,a_batch]
            next_q_val = q_target_model(ns_batch).max(1)[0].detach()
            q_target = r_batch + dqn_gamma*next_q_val*(1-term_batch)
            
            loss = q_model.criterion(q_val,q_target.detach())

            q_model.optimizer.zero_grad()
            loss.backward()
            q_model.optimizer.step()
            q_model.optimizer.zero_grad()

        if n%target_update == 0:
            q_target_model.load_state_dict(q_model.state_dict())
        
        
        epsilon -= eps_decay
    
    return q_model

def test(q_model, config):
    epsilon = 0
    env = gym.make(config['env'])
    env = FullyObsWrapper(env)

    for e in range(2):
        done = False
        state = env.reset()
        n_states = np.product(env.observation_space['image'].shape)
        s = state['image'].reshape(n_states)
        while not done:
            a = q_model.action(s,epsilon)
            next_state,r,done,_ = env.step(a)
            env.render()
            ns = next_state['image'].reshape(n_states)
            s = ns

def model_pipeline(hyperparameters):
    with wandb.init(project="dqn-demo",config=hyperparameters):
        config=wandb.config
        q_model, q_target_model, replay_buffer = make(config)
        q_model = train(q_model,q_target_model, replay_buffer, config)
        test(q_model,config)
    return q_model

model = model_pipeline(config)
