from f110_env import F110Env

# ./build/sim_server '0.01' '2' '6666' '0.523' '0.074' '0.17145' '4.718' '5.4562' '0.04712' '3.74'
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.sac import MlpPolicy

env = F110Env()
env = VecFrameStack(env, 10)
print(env.action_space)
print(env.observation_space)
# loading the map (uses the ROS convention with .yaml and an image file)
map_path = 'maps/berlin.yaml'
map_img_ext = '.png'  # png extension for example
executable_dir = './build/'

# loading physical parameters of the car
# These could be identified on your own system
mass = 3.74
l_r = 0.17145
I_z = 0.04712
mu = 0.523
h_cg = 0.074
cs_f = 4.718
cs_r = 5.4562

env.init_map(map_path, map_img_ext, False, False)
env.update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass, executable_dir)

# Initial state (for two cars)
lap_time = 0.0

# Resetting the environment
obs = env.reset()
print(obs)

sac = SAC(env=env, policy=MlpPolicy, buffer_size=20000, learning_starts=0, train_freq=20000, batch_size=256, verbose=0, gradient_steps=100, learning_rate=0.0005)
while True:
  sac.learn(20000)
  sac.save("sac/model_sb3")

# Simulation loop
# while not done:

# # Your agent here
# ego_speed, opp_speed, ego_steer, opp_steer = agent.plan(obs)
#
# # Stepping through the environment
# action = {'ego_idx': 0, 'speed': [ego_speed, opp_speed], 'steer': [ego_steer, opp_steer]}
# obs, step_reward, done, info = racecar_env.step(action)
#
# # Getting the lap time
# lap_time += step_reward
