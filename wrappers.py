import gym 
import gym_minigrid
from gym import spaces

class MDPWrapper(gym.ObservationWrapper):
    
    def __init__(self,env):
        super().__init__(env)
        imgSpace = env.observation_space.spaces['image'].shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSpace[0]*imgSpace[1],),
            dtype='uint8'
        )

    def observation(self,obs):
        state  = obs['image'][:,:,0].reshape(-1)
        return state