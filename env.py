import gym
from gym import spaces
from gym.spaces import Box, Tuple, MultiDiscrete
import numpy as np
import random
import librosa
import librenderman as rm
from scipy.io.wavfile import write

class VSTEnv(gym.Env):
    def __init__(self, config, vst_config):
        super(VSTEnv, self).__init__()

        self.config = config
        self.vst_config = vst_config

        self.num_knobs = len(vst_config['rnd'])
        self.num_audio_samples = int(config['sampleRate'] * config['renderLength']) # Keep audio samples divisible by fftSize
        self.num_audio_samples = self.num_audio_samples - (self.num_audio_samples % config['fftSize'])
        self.num_freq = int(1+(config['fftSize']/2.0))
        self.num_mfcc = 20
        #self.num_windows = int((self.num_audio_samples / config['fftSize'] - 1.0) * (config['fftSize'] / config['hopSize']) + 1.0)
        self.num_windows = int((self.num_audio_samples / config['fftSize']) * (config['fftSize'] / config['hopSize']) + 1.0)

        # Mapping from action index (0, 1, ..., num_knobs) to VST parameter
        self.action_to_param = list(vst_config['rnd'].keys())

        self.action_space = MultiDiscrete([self.num_knobs, 4])
        #self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_freq, self.num_windows,))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_mfcc, self.num_windows))
        #self.observation_space = spaces.Box(low=0, high=255, shape=(self.num_freq, self.num_windows, 1))

        # Create VST engine and generator
        self.engine = rm.RenderEngine(config['sampleRate'], config['bufferSize'], config['fftSize'])
        self.engine.load_plugin(vst_config['vstPath'])
        self.generator = rm.PatchGenerator(self.engine)

    # Init VST engine params from 'self.vst_config'
    def init_engine(self):
        knob_state = {}
        for param, value in self.vst_config['init'].items():
            if param in self.vst_config['rnd']:
                knob_state[param] = value
            self.engine.override_plugin_parameter(param, value)
        return knob_state

    # Randomize specific set of params as defined in 'self.vst_config'
    def randomize_knobs(self):
        knob_state = {}
        for param, value in self.vst_config['rnd'].items():
            rnd = random.uniform(value[0], value[1])
            rnd = rnd - rnd % 0.01
            knob_state[param] = rnd
        return knob_state

    # Render an audio patch, safe to 'self.cur_audio'
    def render_audio(self, knob_state):
        for param, value in knob_state.items():
            self.engine.override_plugin_parameter(param, value)
        self.engine.render_patch(self.config['midiNote'], self.config['midiVelocity'], self.config['noteLength'], self.config['renderLength'])
        audio = np.array(self.engine.get_audio_frames())[:self.num_audio_samples]
        return audio

    # Compute stft of audio patch, safe to 'self.cur_stft'
    def compute_stft(self, audio):
        return self.compute_mfcc(audio)
        stft = np.abs(librosa.stft(y=audio, hop_length=self.config['hopSize'], center=False))
        stft = (stft - np.min(stft)) / np.max(stft) # Normalize absolute amplitudes to value range [0,1]
        return stft
    def compute_mfcc(self, audio):
        mfcc = librosa.feature.mfcc(y=audio, sr=self.config['sampleRate'])
        '''
        from PIL import Image
        import time
        mat = np.copy(mfcc)
        mat = mat + np.abs(np.min(mat))
        mat = mat / np.max(mat)
        img = Image.fromarray(np.uint8(mat*255), 'L')
        img.show()
        img.save('out.jpg')
        exit()
        '''
        #mfcc_mvn = (mfcc - np.expand_dims(np.mean(mfcc, axis=1), axis=1))/np.expand_dims(np.std(mfcc, axis=1), axis=1)
        #mfcc_msn = (mfcc - np.expand_dims(np.mean(mfcc, axis=1), axis=1))
        #mfcc_msn /= np.max(np.abs(mfcc_msn))
        mfcc_sn = mfcc/np.max(np.abs(mfcc))
        return mfcc_msn

    def compute_reward(self):
        # similarity := 1/(1+dist)
        #similarity = lambda x, y: 1/(1+(4*np.linalg.norm(self.target_stft - self.cur_stft)/(4.0*self.num_freq*self.num_windows)))
        self.stft_dif = self.goal_stft - self.stft

        if not hasattr(self, 'similarity'):
            self.similarity = 1/(1+np.linalg.norm(self.stft_dif))
        else:
            self.old_similarity = self.similarity
            self.similarity = 1/(1+np.linalg.norm(self.stft_dif))
            self.reward = float(self.similarity - self.old_similarity)
            self.done = self.similarity > 0.95

        #if self.cur_step % 100 == 0:
        #z = set(self.knob_state) & set(self.target_knobs)
        #asd = ''
        #for i in z:
        #    asd = asd + str(i) + ": (" + '{:0.2f}'.format(self.knob_state[i]) + " / " + '{:0.2f}'.format(self.target_knobs[i]) + "), "
        #print(self.cur_step, asd, self.cur_similarity, self.reward)

        #if self.done:
        #    z = set(self.knob_state) & set(self.target_knobs)
        #    asd = '(Done) - '
        #    for i in z:
        #        asd = asd + str(i) + ": (" + '{:0.2f}'.format(self.knob_state[i]) + " / " + '{:0.2f}'.format(self.target_knobs[i]) + "), "
        #    print(self.cur_step, asd, self.cur_similarity, self.reward)

    def step(self, action):
        knob = action[0]
        amount = 0.1 if action[1] % 2 == 0 else 0.05
        amount = -amount if action[1] < 2 else amount

        # Apply knob adjustments, clip VST param to [0, 1.0]
        clip = lambda x, l, u: max(l, min(u, x))
        #self.knob_state[self.action_to_param[action[0]]] = clip(self.knob_state[self.action_to_param[action[0]]] + action[1][0], 0, 1.0)
        self.knob_state[self.action_to_param[knob]] = clip(self.knob_state[self.action_to_param[knob]] + amount, 0, 1.0)

        # Render audio, compute stft and reward
        self.audio= self.render_audio(self.knob_state)
        self.stft = self.compute_stft(self.audio)
        self.compute_reward()

        #observation = np.expand_dims((self.stft - self.goal_stft + 1.0) * 128.0, axis=2)
        observation = self.stft - self.goal_stft
        info = {}

        #if self.cur_step % 100 == 0:
        #    write(str(self.cur_step) + '.wav', self.config['sampleRate'], self.cur_audio)
        return observation, self.reward, self.done, info

    def reset(self):
        self.iter = 0

        # Create goal
        _ = self.init_engine() # Init engine params
        self.goal_knob_state = self.randomize_knobs() # Randomize engine params
        self.goal_audio = self.render_audio(self.goal_knob_state) # Render audio
        self.goal_stft = self.compute_stft(self.goal_audio) # Compute stft
        #write('target.wav', self.config['sampleRate'], self.target_audio)

        # Reinitialize VST engine to start state
        self.knob_state = self.init_engine()
        self.audio = self.render_audio(self.knob_state)
        self.stft = self.compute_stft(self.audio)
        #self.mfcc = self.compute_mfcc(self.audio)
        #print(self.mfcc)
        #print(self.mfcc.shape)
        #print(np.min(self.mfcc))
        #print(np.max(self.mfcc))
        #print(np.sum(self.mfcc))

        # Compute similarity between target and start
        self.compute_reward()

        #observation = np.expand_dims((self.cur_stft - self.target_stft + 1.0) * 128.0, axis=2)
        observation = self.stft - self.goal_stft
        return observation

    def render(self, mode='console'):
        print('Metric:', self.reward)

    def close(self):
        pass
