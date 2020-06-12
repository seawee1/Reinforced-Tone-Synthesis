import yaml
import os
import gym
from env import VSTEnv
from stable_baselines.common.env_checker import check_env
#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C, DQN


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


# Load configs, import librenderman
config, config_vst = load_configs()
symlink_librenderman(config)

# Create name
env = VSTEnv(config, config_vst)
#model = A2C(MlpPolicy, env, verbose=1)
#model.learn(total_timesteps=25000)
model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000)
model.save('test1')
#check_env(env, warn=True)
