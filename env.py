import gym
from gym import spaces
from gym.spaces import Box, Tuple, Discrete
import numpy as np
import random
import librosa
import librenderman as rm

class VSTEnv(gym.Env):
    def __init__(self, config, vst_config):
        super(VSTEnv, self).__init__()

        self.config = config
        self.vst_config = vst_config

        # Number of VST knobs to operate
        self.num_knobs = len(vst_config['rnd'])
        self.num_audio_samples = int(config['sampleRate'] * config['renderLength'])
        self.num_audio_samples = self.num_audio_samples - (self.num_audio_samples % config['fftSize']) # Keep audio samples divisible by fftSize
        self.num_freq = int(1+(config['fftSize']/2.0))
        self.num_windows = int((self.num_audio_samples / config['fftSize'] - 1.0) * (config['fftSize'] / config['hopSize']) + 1.0)

        # Mapping from action index (0, 1, ..., num_knobs) to VST parameter
        #self.action_to_param_dict = dict(enumerate(vst_config['rnd'].keys()))
        self.action_to_param = list(vst_config['rnd'].keys())


        ## For now, operate all knobs at once
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_knobs,))
        #self.action_space = Tuple((
        #    Discrete(self.num_knobs),
        #    Box(low=-1.0, high=1.0, shape=(1,))
        #))
        self.action_space = Box(low=0.0, high=self.num_knobs*2.0, shape=(1,))

        # observation = state of VST knobs + rendered audio patch
        #self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_knobs + self.num_audio_samples,))
        #self.observation_space = spaces.Tuple((
        #    spaces.Box(low=0.0, high=1.0, shape=(self.num_knobs,)),
        #    spaces.Box(low=0.0, high=1.0, shape=(self.num_freq, self.num_windows)),
        #    spaces.Box(low=0.0, high=1.0, shape=(self.num_freq, self.num_windows))
        #))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_knobs + 2*self.num_freq*self.num_windows,))

        # Create VST engine and generator
        self.engine = rm.RenderEngine(config['sampleRate'], config['bufferSize'], config['fftSize'])
        self.engine.load_plugin(vst_config['vstPath'])
        self.generator = rm.PatchGenerator(self.engine)

        # Other self params
        self.knob_state = {}
        self.cur_audio = None
        self.cur_stft = None
        self.target_audio = None
        self.target_stft = None
        self.cur_metric = 0.0
        self.old_metric = 0.0


    # Init VST engine params from 'self.vst_config'
    def init_engine(self):
        for param, value in self.vst_config['init'].items():
            if param in self.vst_config['rnd']:
                self.knob_state[param] = value
            self.engine.override_plugin_parameter(param, value)

    # Randomize specific set of params as defined in 'self.vst_config'
    def randomize_engine(self):
        for param, value in self.vst_config['rnd'].items():
            self.knob_state[param] = random.uniform(value[0], value[1])
        self.set_engine()

    # Set params (the ones defined to be randomized) to specific values specified in 'self.state'
    def set_engine(self):
        for param, value in self.knob_state.items():
            self.engine.override_plugin_parameter(param, value)

    # Render an audio patch, safe to 'self.cur_audio'
    def render_patch(self):
        self.set_engine()
        self.engine.render_patch(self.config['midiNote'], self.config['midiVelocity'], self.config['noteLength'], self.config['renderLength'], True)
        self.cur_audio = np.array(self.engine.get_audio_frames())[:self.num_audio_samples]

    # Compute stft of audio patch, safe to 'self.cur_stft'
    def compute_stft(self):
        self.cur_stft = np.abs(librosa.stft(y=self.cur_audio, center=False))
        self.cur_stft = (self.cur_stft - np.min(self.cur_stft)) / np.max(self.cur_stft) # Normalize absolute amplitudes to value range [0,1]

    #def compute_similarity(self):
    #    gaussian_kernel = lambda x, y, sigma: np.exp((-(np.linalg.norm(x-y)))/(2*sigma**2))
    #    self.old_similarity = self.cur_similarity
    #    self.cur_similarity = gaussian_kernel(target_audio, cur_audio, 5.0)

    def compute_reward(self):
        self.old_metric = self.cur_metric

        self.frobenius_matrix = np.square(np.abs(self.target_stft - self.cur_stft))
        self.cur_metric = float(np.sqrt(np.sum(self.frobenius_matrix)))

        self.reward = float(self.old_metric - self.cur_metric)
        if(self.reward==0.0):
            self.reward = -1.0
        print(self.cur_metric, self.reward)
        self.done = bool(np.all(self.frobenius_matrix < self.config['epsilon']**2))

    def step(self, action):
        self.cur_action = action
        knob = int(action[0] // 2.0)
        amount = (action[0] % 2.0) - 1.0

        # Apply knob adjustments, clip VST param to [0, 1.0]
        clip = lambda x, l, u: max(l, min(u, x))
        #self.knob_state[self.action_to_param[action[0]]] = clip(self.knob_state[self.action_to_param[action[0]]] + action[1][0], 0, 1.0)
        self.knob_state[self.action_to_param[knob]] = clip(self.knob_state[self.action_to_param[knob]] + amount, 0, 1.0)


        # Render audio, compute stft and reward
        self.set_engine()
        self.render_patch()
        self.compute_stft()
        self.compute_reward()

        observation = np.concatenate((np.array(list(self.knob_state.values())).flatten(), self.target_stft.flatten(), self.cur_stft.flatten()))
        info = {}
        return observation, self.reward, self.done, info

    def reset(self):
        # Create random target
        self.init_engine() # Init engine params
        self.randomize_engine() # Randomize engine params
        self.render_patch() # Render audio
        self.target_audio = np.copy(self.cur_audio)
        self.compute_stft() # Compute stft
        self.target_stft = np.copy(self.cur_stft)

        # Reinitialize VST engine to start state
        self.init_engine()
        self.render_patch()
        self.compute_stft()

        # Compute similarity between target and start
        self.compute_reward()

        observation = np.concatenate((np.array(list(self.knob_state.values())).flatten(), self.target_stft.flatten(), self.cur_stft.flatten()))
        return observation

    def render(self, mode='console'):
        print('Metric:', self.reward)

    def close(self):
        pass
