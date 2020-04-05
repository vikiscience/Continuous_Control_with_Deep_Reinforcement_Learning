from pathlib import Path

file_name_env = 'D:\D_Downloads\Reacher_Windows_x86_64\Reacher.exe'

model_path = Path('./models/')
output_path = Path('./output/')
file_path_model = model_path / 'model.npy'
file_path_ref_model = model_path / 'model_ref.npy'
file_path_img_score = output_path / 'score.png'
file_path_ref_img_score = output_path / 'score_ref.png'

# general params
random_seed = 0xABCD
rolling_mean_N = 100
num_agents = 1
state_size = 33
action_size = 4
max_action = 1.
episode_length = 300
verbose = False

# algo params
num_episodes = 2000
policy_eval_freq = 10

# agent params
memory_size = 20000
gamma = 0.95
batch_size = 128
expl_noise = 0.1

# model params
model_learning_rate = 0.0001
model_fc1_num = 32
model_fc2_num = 16


def myprint(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)
