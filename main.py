import yaml
import os
import gym
from env import VSTEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
#from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C, DQN, PPO2
from stable_baselines.common.callbacks import BaseCallback
import tensorflow as tf


#from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.cmd_util import make_vec_env

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)

    def summary_images(self, tag, images):
        from io import BytesIO ## for Python 3
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import cm, imsave

        im_summaries = []
        for nr, img in enumerate(images):
        # Write the image to a string
            s = BytesIO()
            imsave(s, img, format='png', cmap=cm.coolwarm)

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(), height=img.shape[0], width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        return summary

    def _on_step(self):
        if not self.is_tb_set:
            #print(type(self.model.get_env().get_attr('goal_stft')[0]))
            #print((self.model.get_env().get_attr('goal_stft'))[0].shape)
            with self.model.graph.as_default():

                self.model.summary = tf.summary.merge_all()
            self.is_tb_set = True

        #stft_to_tensor = lambda attr_name: tf.expand_dims(tf.convert_to_tensor(self.model.get_env().get_attr(attr_name)[0]), 2)
        #img = tf.stack([stft_to_tensor('goal_stft'), stft_to_tensor('stft'), stft_to_tensor('stft_dif')])
        images = [self.model.get_env().get_attr('goal_stft')[0], self.model.get_env().get_attr('stft')[0], self.model.get_env().get_attr('stft_dif')[0]]
        summary = self.summary_images('stft', images)

        self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True

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

def symlink_librenderman(config):
    # Symlink RenderMan library to working directory before import
    if not os.path.isfile('librenderman.so'):
        os.symlink(config['libPath'], os.path.join('.', 'librenderman.so'))
    import librenderman as rm

def make_env(config, config_vst, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = VSTEnv(config, config_vst)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


# Load configs, import librenderman
config, config_vst = load_configs()
symlink_librenderman(config)

"""
num_cpu = 4
env = SubprocVecEnv([make_env(config, config_vst, i) for i in range(num_cpu)])
model = PPO2(CnnPolicy, env)
model.learn(total_timesteps=50000)
model.save('ppo2_test')
"""
# Create name
env = VSTEnv(config, config_vst)
#model = A2C(MlpPolicy, env, verbose=1)
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='tensorboard/')
model.learn(total_timesteps=50000, callback=TensorboardCallback(env))
model.save('ppo2_test')
#check_env(env, warn=True#)
