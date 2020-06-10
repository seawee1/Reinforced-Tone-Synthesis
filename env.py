import gym
from gym import spaces
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
        self.num_mfcc = math.ceil(config['sampleRate']*config['renderLength']/config['fftSize'])
        # Mapping from action index (0, 1, ..., num_knobs) to VST parameter
        self.action_to_param_dict = dict(enumerate(vst_config['rnd'].keys()))

        ## For now, operate all knobs at once
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_knobs,))

        self.action_space = spaces.Tuple(spaces.Discrete(self.num_knobs), spaces.Box(low=-1.0, high=1.0, shape=(self.num_knobs,))
        # observation = state of VST knobs + rendered audio patch
        #self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_knobs + self.num_audio_samples,))
        self.observation_space = spaces.dict({
                                              'knob_state': spaces.Box(low=0.0, high=1.0, shape=(self.num_knobs)),
                                              'target_mfcc': spaces.Box(low=0.0, high=1.0, shape=(config['mfccSize'], self.num_mfcc)),
                                              'cur_mfcc': spaces.Box(low=0.0, high=1.0, shape=(config['mfccSize'], self.num_mfcc)),
                                               })

        # Create VST engine and generator
        self.engine = rm.RenderEngine(config['sampleRate'], config['bufferSize'], config['fftSize'])
        self.engine.load_plugin(vst_config['vstPath'])
        self.generator = rm.PatchGenerator(self.engine)

        # Other self params
        self.knob_state = {}
        self.cur_audio = None
        self.cur_mfccs = None
        self.target_audio = None
        self.target_mfccs = None
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

    # Compute mfccs of audio patch, safe to 'self.cur_mfccs'
    def compute_mfccs(self):
        self.cur_mfccs = librosa.feature.mfcc(y=self.cur_audio, sr=self.config['sampleRate'], n_mfcc = self.config['mfccSize'])

    #def compute_similarity(self):
    #    gaussian_kernel = lambda x, y, sigma: np.exp((-(np.linalg.norm(x-y)))/(2*sigma**2))
    #    self.old_similarity = self.cur_similarity
    #    self.cur_similarity = gaussian_kernel(target_audio, cur_audio, 5.0)

    def compute_reward(self):
        self.old_metric = self.cur_metric
        self.cur_metric = np.linalg.norm(self.target_mfccs - self.cur_mfccs)
        self.reward = float(self.old_metric - self.cur_metric)
        self.done = (self.reward < 1.0)

    def step(self, action):
        self.cur_action = action

        # Apply knob adjustments, clip VST param to [0, 1.0]
        clip = lambda x, l, u: max(l, min(u, x))
        for i, x in enumerate(action):
            self.knob_state[self.action_to_param_dict[i]] = clip(self.knob_state[self.action_to_param_dict[i]] + x, 0, 1.0)

        # Render audio and mfccs, compute feature metric
        self.set_engine()
        self.render_patch()
        self.compute_mfccs()
        self.compute_reward()
        print(self.cur_metric)
        print(action)

        observation = np.concatenate([np.array(list(self.knob_state.values())), self.cur_audio])
        info = {}
        return observation, self.reward, self.done, info

    def reset(self):
        # Create random target
        self.init_engine() # Init engine params
        self.randomize_engine() # Randomize engine params
        self.render_patch() # Render audio
        self.target_audio = np.copy(self.cur_audio)
        self.compute_mfccs() # Compute mfccs
        self.target_mfccs = np.copy(self.cur_mfccs)
        print(self.cur_audio.shape)
        print(np.max(self.target_mfccs))

        # Reinitialize VST engine to start state
        self.init_engine()
        self.render_patch()
        self.compute_mfccs()

        # Compute similarity between target and start
        self.compute_reward()

        observation = np.concatenate([np.array(list(self.knob_state.values())), self.cur_audio])
        return observation

    def render(self, mode='console'):
        print('Metric:', self.reward)

    def close(self):
        pass
