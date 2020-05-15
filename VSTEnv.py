import gym
from gym import spaces

class VSTEnv(gym.Env):
    def __init__(self, config, vst_config):
        super(VSTEnv, self).__init__()

        self.config = config
        self.vst_config = vst_config

        self.num_knobs = len(vst_config['rnd'])
        self.obs_param_dict = dict(enumerate(vst_config['rnd'].keys()))
        self.observation_space = spaces.Tuple(spaces.Discrete())



