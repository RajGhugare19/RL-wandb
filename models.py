import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0")

class DQNModel(nn.Module):
    
    def __init__(self,config):
          
        super(DQNModel, self).__init__()

        self.n_actions = config['action_size']
        self.n_states = config['state_size']

        self.linear1 = nn.Linear(self.n_states,16)
        self.linear2 = nn.Linear(16,self.n_actions)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(),lr = config['learning_rate'])

    def forward(self, x):
     
        q = torch.relu(self.linear1(x.float()))  
        q_value = self.linear2(q)

        return q_value
    
    def action(self, x, epsilon):

        r = np.random.random()
        if r < epsilon:
            action = np.random.randint(0,self.n_actions)

        else:
            with torch.no_grad():
                x = torch.tensor(x).to(device)
                q_val = self.forward(x)
                action = torch.argmax(q_val).item()
        return action


        x = self.linear2(x)

        return x
