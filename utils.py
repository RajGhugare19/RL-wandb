import numpy as np

class DQNMemory():

    def __init__(self,config):

        self.batch_size = config['batch_size'] 
        self.memory_size = config['memory_size']
        
        self.state_memory_size  = [config['memory_size'],config['state_size']]       
        
        self.memory = []
        self.memory_count = 0

    def store(self,s,a,r,t,ns):
        
        transition = (s,a,r,t,ns)
        
        if self.memory_count>=self.memory_size:
            index = self.memory_count%self.memory_size
            self.memory[index] = transition
        else:
            self.memory.append(transition)

        self.memory_count+=1
    
    def sample(self):
        s,a,r,t,ns = [],[],[],[],[]
        if self.memory_count>self.memory_size:
            ch = self.memory_size
        else:
            ch = self.memory_count
        batch = np.random.choice(ch, self.batch_size, replace=True)    
        for ind in batch:
            sample = self.memory[ind]
            s_,a_,r_,t_,ns_ = sample
            s.append(s_)
            a.append(a_)
            r.append(r_)
            t.append(t_)
            ns.append(ns_)

        return np.array(s,dtype=np.float32), np.array(a,dtype=np.int), np.array(r,dtype=np.float32), np.array(t,dtype=np.int8), np.array(ns,dtype=np.float32)
