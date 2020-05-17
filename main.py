import yaml
import os
import random

import gym
from gym import spaces
import numpy as np
import librosa

# Symlink RenderMan library to working directory before import
if not os.path.isfile('librenderman.so'):
    os.symlink(config[libPath], os.path.join('.', 'librenderman.so'))

import librenderman as rm
class VSTEnv(gym.Env):
    def __init__(self, config, vst_config):
        super(VSTEnv, self).__init__()
        self.config = config
        self.vst_config = vst_config

        # Number of VST knobs to operate
        self.num_knobs = len(vst_config['rnd'])
        # Mapping from knob index to VST param
        #self.obs_param_dict = dict(enumerate(vst_config['rnd'].keys()))

        # For now, operate all knobs at once
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_knobs,))
        # observation = state of VST knobs + rendered audio patch
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_knobs + config['sampleRate'] * config['renderLength']))

        # Create VST engine and generator
        self.engine = rm.RenderEngine(config['sampleRate'], config['bufferSize'], config['fftSize'])
        self.engine.load_plugin(config_vst['vstPath'])
        self.generator = rm.PatchGenerator(engine)
        # Initialize and randomize VST params
        self.init_engine() # Load init params
        self.state = self.randomize_engine() # Randomize params and save the state

    # Load init params from config_vst
    def init_engine(self):
        for param, value in self.config_vst['init'].items():
            self.engine.override_plugin_parameter(param, value)

    # Randomize specific set of params as defined in config_vst
    def randomize_engine(self):
        state = {}
        for param, value in self.config_vst['rnd'].items():
            state[param] = random.uniform(value[0], value[1])
            self.engine.override_plugin_parameter(param, state[param])
        return state

    # Set params (the ones defined to be randomized) to specific values
    def set_engine(self, state):
        for param, value in state.items():
            self.engine.override_plugin_parameter(param, state[param])

    # Render an audio patch
    def render_patch(self):
        self.engine.render_patch(self.config['midiNote'], self.config['midiVelocity'], self.config['noteLength'], self.config['renderLength'], True)
        audio = engine.get_audio_frames()
        #mfccs = engine.get_mfcc_frames()
        return np.array(audio) #, mfccs

    # Compute mfccs of audio patch
    def compute_mfccs(self, audio):
        mfccs = librosa.feature.mfcc(y=audio, sr=config['sampleRate'], n_mfcc = 13)
        return mfccs

    def step(self, action):
        # TODO
        return observation, reward, done, info

    def reset(self):
        # TODO
        return observation


def load_configs():
    with open("./config.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    with open(os.path.join('vst_config', config['vst'] + '.yaml'), 'r') as stream:
        try:
            config_vst = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config, config_vst



config, config_vst = load_configs()
env = VSTEnv(config, config_vst)
