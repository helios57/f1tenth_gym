import os
import signal
import sys
from math import fabs, sqrt

import numpy as np
import yaml
import zmq
from PIL import Image
from gym import utils, spaces

import sim_requests_pb2
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.util import obs_space_info


class F110Env(VecEnv, utils.EzPickle):
  """
  OpenAI gym environment for F1/10 simulator
  Use 0mq's REQ-REP pattern to communicate to the C++ simulator
  ONE env has ONE corresponding C++ instance
  Need to create env with map input, full path to map yaml file, map pgm image and yaml should be in same directory

  should be initialized with a map, a timestep
  """
  metadata = {'render.modes': []}

  def __init__(self):
    # self.stackSize = 10
    self.scanSize = 1080
    self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
    # stacked self.stackSize latest observations of (scan, x,y,theta,vel_x, vel_y, vel_z)*2 (we get the data from the opponent as well)
    self.observation_space = spaces.Box(low=0.0, high=1.0, shape=((self.scanSize + 6) * 2,), dtype=np.float32)
    VecEnv.__init__(self, 2, self.observation_space, self.action_space)
    self.keys, shapes, dtypes = obs_space_info(self.observation_space)

    self.actions = None

    # simualtor params
    self.params_set = False
    self.map_inited = False
    # params list is [mu, h_cg, l_r, cs_f, cs_r, I_z, mass]
    self.params = []
    self.num_agents = 2
    self.timestep = 0.01
    self.map_path = None
    self.map_img = None
    self.ego_idx = 0
    self.timeout = 120.0
    # radius to consider done
    self.start_thresh = 0.5  # 10cm
    # env states
    # more accurate description should be ego car state
    # might not need to keep scan
    self.x = None
    self.y = None
    self.theta = None

    self.in_collision = False
    self.collision_angle = None

    # loop completion
    self.near_start = True
    self.num_toggles = 0

    # race info
    self.lap_times = [0.0, 0.0]
    self.lap_counts = [0, 0]

    self.map_height = 0.0
    self.map_width = 0.0
    self.map_resolution = 0.0
    self.free_thresh = 0.0
    self.origin = []
    self.port = 6666
    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.PAIR)
    self.socket.connect('tcp://localhost:6666')
    print('Gym env - Connected env to port: ' + str(self.port))
    self.sim_p = None
    print('Gym env - env created, waiting for params...')

  def step_async(self, actions: np.ndarray):
    self.actions = actions

  def step_wait(self):
    return self.step(self.actions)

  def __del__(self):
    """
    Finalizer, does cleanup
    """
    if self.sim_p is None:
      pass
    else:
      os.kill(self.sim_p.pid, signal.SIGTERM)
      # print('Gym env - Sim child process killed.')

  def _start_executable(self, path):
    mu = self.params[0]
    h_cg = self.params[1]
    l_r = self.params[2]
    cs_f = self.params[3]
    cs_r = self.params[4]
    I_z = self.params[5]
    mass = self.params[6]
    # mass= 3.74
    # l_r = 0.17145
    # I_z = 0.04712
    # mu = 0.523
    # h_cg = 0.074
    # cs_f = 4.718
    # cs_r = 5.4562
    args = [path + 'sim_server', str(self.timestep), str(self.num_agents), str(self.port), str(mu), str(h_cg), str(l_r), str(cs_f), str(cs_r), str(I_z), str(mass)]
    print("Start", args)
    # self.sim_p = subprocess.Popen(args)

  def _set_map(self):
    """
    Sets the map for the simulator instance
    """
    if not self.map_inited:
      print('Gym env - Sim map not initialized, call env.init_map() to init map.')
    # create and fill in protobuf
    map_request_proto = sim_requests_pb2.SimRequest()
    map_request_proto.type = 1
    map_request_proto.map_request.map.extend((1. - self.map_img / 255.).flatten().tolist())
    map_request_proto.map_request.origin_x = self.origin[0]
    map_request_proto.map_request.origin_y = self.origin[1]
    map_request_proto.map_request.map_resolution = self.map_resolution
    # TODO: double check if this value is valid
    map_request_proto.map_request.free_threshold = self.free_thresh
    map_request_proto.map_request.map_height = self.map_height
    map_request_proto.map_request.map_width = self.map_width
    # serialization
    map_request_string = map_request_proto.SerializeToString()
    # send set map request
    print('Gym env - Sending set map request...')
    self.socket.send(map_request_string)
    print('Gym env - Map request sent.')
    # receive response from sim instance
    sim_response_string = self.socket.recv()
    # parse map response proto
    sim_response_proto = sim_requests_pb2.SimResponse()
    sim_response_proto.ParseFromString(sim_response_string)
    # get results
    set_map_result = sim_response_proto.map_result.result
    if set_map_result == 1:
      print('Gym env - Set map failed, exiting...')
      sys.exit()

  def _update_state(self, obs_dict):
    """
    Update the env's states according to observations
    obs is observation dictionary
    """
    self.x = obs_dict['poses_x'][obs_dict['ego_idx']]
    self.y = obs_dict['poses_y'][obs_dict['ego_idx']]

    self.theta = obs_dict['poses_theta'][obs_dict['ego_idx']]
    self.in_collision = obs_dict['collisions'][obs_dict['ego_idx']]
    self.collision_angle = obs_dict['collision_angles'][obs_dict['ego_idx']]

  def step(self, action):
    # can't step if params not set
    if not self.params_set:
      print('ERROR - Gym Env - Params not set, call update params before stepping.')
      sys.exit()
    # action is a list of steering angles + command velocities
    # also a ego car index
    # action should a DICT with {'ego_idx': int, 'speed':[], 'steer':[]}
    step_request_proto = sim_requests_pb2.SimRequest()
    step_request_proto.type = 0
    step_request_proto.step_request.ego_idx = 0
    step_request_proto.step_request.requested_vel.extend(action[0])
    step_request_proto.step_request.requested_ang.extend(action[1])
    # serialization
    step_request_string = step_request_proto.SerializeToString()
    # send step request
    self.socket.send(step_request_string)
    # receive response from sim instance
    sim_response_string = self.socket.recv()
    # print('Gym env - Received response for step request.')
    # parse map response proto
    sim_response_proto = sim_requests_pb2.SimResponse()
    sim_response_proto.ParseFromString(sim_response_string)
    # get results
    # make sure we have the right type of response
    response_type = sim_response_proto.type
    # TODO: also check for stepping fail
    if not response_type == 0:
      print('Gym env - Wrong response type for stepping, exiting...')
      sys.exit()
    observations_proto = sim_response_proto.sim_obs
    # make sure the ego idx matches
    if not observations_proto.ego_idx == 0:
      print('Gym env - Ego index mismatch, exiting...')
      sys.exit()
    # get observations
    carobs_list = observations_proto.observations
    # construct observation dict
    # Observation DICT, assume indices consistent: {'ego_idx':int, 'scans':[[]], 'poses_x':[], 'poses_y':[], 'poses_theta':[], 'linear_vels_x':[], 'linear_vels_y':[], 'ang_vels_z':[], 'collisions':[], 'collision_angles':[]}
    simulation_input = {'ego_idx': observations_proto.ego_idx, 'scans': [], 'poses_x': [], 'poses_y': [], 'poses_theta': [], 'linear_vels_x': [], 'linear_vels_y': [], 'ang_vels_z': [],
                        'collisions': [],
                        'collision_angles': [], 'lap_times': [], 'lap_counts': []}
    for car_obs in carobs_list:
      simulation_input['scans'].append(car_obs.scan)
      simulation_input['poses_x'].append(car_obs.pose_x)
      simulation_input['poses_y'].append(car_obs.pose_y)
      if abs(car_obs.theta) < np.pi:
        simulation_input['poses_theta'].append(car_obs.theta)
      else:
        simulation_input['poses_theta'].append(-((2 * np.pi) - car_obs.theta))
      simulation_input['linear_vels_x'].append(car_obs.linear_vel_x)
      simulation_input['linear_vels_y'].append(car_obs.linear_vel_y)
      simulation_input['ang_vels_z'].append(car_obs.ang_vel_z)
      simulation_input['collisions'].append(car_obs.collision)
      simulation_input['collision_angles'].append(car_obs.collision_angle)

    simulation_input['lap_times'] = self.lap_times
    simulation_input['lap_counts'] = self.lap_counts
    # update accumulated time in env
    self.current_time = self.current_time + self.timestep
    self._update_state(simulation_input)
    obs1 = np.concatenate([
      np.asarray(simulation_input['scans'][0]),
      np.asarray(simulation_input['scans'][1]),
      np.asarray(simulation_input['poses_x']),
      np.asarray(simulation_input['poses_y']),
      np.asarray(simulation_input['poses_theta']),
      np.asarray(simulation_input['linear_vels_x']),
      np.asarray(simulation_input['linear_vels_y']),
      np.asarray(simulation_input['ang_vels_z'])
    ])

    obs2 = np.concatenate([
      np.asarray(simulation_input['scans'][1]),
      np.asarray(simulation_input['scans'][0]),
      np.asarray(np.asarray([simulation_input['poses_x'][1], simulation_input['poses_x'][0]])),
      np.asarray(np.asarray([simulation_input['poses_y'][1], simulation_input['poses_y'][0]])),
      np.asarray(np.asarray([simulation_input['poses_theta'][1], simulation_input['poses_theta'][0]])),
      np.asarray(np.asarray([simulation_input['linear_vels_x'][1], simulation_input['linear_vels_x'][0]])),
      np.asarray(np.asarray([simulation_input['linear_vels_y'][1], simulation_input['linear_vels_y'][0]])),
      np.asarray(np.asarray([simulation_input['ang_vels_z'][1], simulation_input['ang_vels_z'][0]]))
    ])

    reward1 = sqrt(
      simulation_input['linear_vels_x'][0] * simulation_input['linear_vels_x'][0] +
      simulation_input['linear_vels_y'][0] * simulation_input['linear_vels_y'][0]) \
              - simulation_input['collisions'][1] * 10 - fabs(action[1][0]) * 0.1
    reward2 = sqrt(
      simulation_input['linear_vels_x'][1] * simulation_input['linear_vels_x'][1] +
      simulation_input['linear_vels_y'][1] * simulation_input['linear_vels_y'][1]) \
              - simulation_input['collisions'][1] * 10 - fabs(action[1][1]) * 0.1
    done = False
    if simulation_input['collisions'][0]:
      reward1 = -10
      done = True
    if simulation_input['collisions'][1]:
      reward2 = -10
      done = True

    obs = np.array([obs1, obs2])
    reward = np.array([reward1, reward2])
    if done:
      print(reward, simulation_input['lap_times'], simulation_input['lap_counts'])
    done = np.array([done, done])
    info = np.array([{}, {}])
    return obs, reward, done, info

  def reset(self, poses={'x': [0.0, 2.0],
                         'y': [0.0, 0.0],
                         'theta': [0.0, 0.0]}):
    self.current_time = 0.0
    self.in_collision = False
    self.collision_angles = None
    self.num_toggles = 0
    self.near_start = True
    self.near_starts = np.array([True] * self.num_agents)
    self.toggle_list = np.zeros((self.num_agents,))
    if poses:
      pose_x = poses['x']
      pose_y = poses['y']
      pose_theta = poses['theta']
      self.start_x = pose_x[0]
      self.start_y = pose_y[0]
      self.start_theta = pose_theta[0]
      self.start_xs = np.array(pose_x)
      self.start_ys = np.array(pose_y)
      self.start_thetas = np.array(pose_theta)
      self.start_rot = np.array([[np.cos(-self.start_theta), -np.sin(-self.start_theta)],
                                 [np.sin(-self.start_theta), np.cos(-self.start_theta)]])
      # create reset by pose proto
      reset_request_proto = sim_requests_pb2.SimRequest()
      reset_request_proto.type = 4
      reset_request_proto.reset_bypose_request.num_cars = self.num_agents
      reset_request_proto.reset_bypose_request.ego_idx = 0
      reset_request_proto.reset_bypose_request.car_x.extend(pose_x)
      reset_request_proto.reset_bypose_request.car_y.extend(pose_y)
      reset_request_proto.reset_bypose_request.car_theta.extend(pose_theta)
      reset_request_string = reset_request_proto.SerializeToString()
      self.socket.send(reset_request_string)
    else:
      # create reset proto
      self.start_x = 0.0
      self.start_y = 0.0
      self.start_theta = 0.0
      self.start_rot = np.array([[np.cos(-self.start_theta), -np.sin(-self.start_theta)],
                                 [np.sin(-self.start_theta), np.cos(-self.start_theta)]])
      reset_request_proto = sim_requests_pb2.SimRequest()
      reset_request_proto.type = 2
      reset_request_proto.reset_request.num_cars = self.num_agents
      reset_request_proto.reset_request.ego_idx = 0
      # serialize reset proto
      reset_request_string = reset_request_proto.SerializeToString()
      # send reset proto string
      self.socket.send(reset_request_string)
    # receive response from sim
    reset_response_string = self.socket.recv()
    reset_response_proto = sim_requests_pb2.SimResponse()
    reset_response_proto.ParseFromString(reset_response_string)
    if reset_response_proto.reset_resp.result:
      print('Gym env - Reset failed')
      # TODO: failure handling
      return None
    # TODO: return with gym convention, one step?
    action = np.zeros((2, 2), dtype=np.float32)
    # print('Gym env - Reset done')
    obs, reward, done, info = self.step(action)
    # print('Gym env - step done for reset')
    return obs

  def init_map(self, map_path, img_ext, rgb, flip):
    """
        init a map for the gym env
        map_path: full path for the yaml, same as ROS, img and yaml in same dir
        rgb: map grayscale or rgb
        flip: if map needs flipping
    """

    self.map_path = map_path
    if not map_path.endswith('.yaml'):
      print('Gym env - Please use a yaml file for map initialization.')
      print('Exiting...')
      sys.exit()

    # split yaml ext name
    map_img_path = os.path.splitext(self.map_path)[0] + img_ext
    self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
    self.map_img = self.map_img.astype(np.float64)
    if flip:
      self.map_img = self.map_img[::-1]

    if rgb:
      self.map_img = np.dot(self.map_img[..., :3], [0.29, 0.57, 0.14])

    # update map metadata
    self.map_height = self.map_img.shape[0]
    self.map_width = self.map_img.shape[1]
    self.free_thresh = 0.6  # TODO: double check
    with open(self.map_path, 'r') as yaml_stream:
      try:
        map_metadata = yaml.safe_load(yaml_stream)
        self.map_resolution = map_metadata['resolution']
        self.origin = map_metadata['origin']
      except yaml.YAMLError as ex:
        print(ex)
    self.map_inited = True

    # load waypoints
    # self.csv_path = os.path.splitext(self.map_path)[0] + '.csv'
    # with open(self.csv_path) as f:
    #     self.waypoints = [tuple(line) for line in csv.reader(f)]
    #     # waypoints are [x, y, speed, theta]
    #     self.waypoints = np.array([(float(pt[0]), float(pt[1]), float(pt[2]), float(pt[3])) for pt in self.waypoints])

  def render(self, mode='human', close=False):
    return

  # def get_min_dist(self, position):
  #     wpts = self.waypoints[:, 0:2]
  #      # = position[0:2]
  #     nearest_point, nearest_dist, t, i = self.nearest_point_on_trajectory(position, wpts)
  #     # speed = self.waypoints[i, 2]
  #     return nearest_dist

  # def nearest_point_on_trajectory(self, point, trajectory):
  #     '''
  #     Return the nearest point along the given piecewise linear trajectory.

  #     Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
  #     not be an issue so long as trajectories are not insanely long.

  #         Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

  #     point: size 2 numpy array
  #     trajectory: Nx2 matrix of (x,y) trajectory waypoints
  #         - these must be unique. If they are not unique, a divide by 0 error will destroy the world
  #     '''
  #     diffs = trajectory[1:,:] - trajectory[:-1,:]
  #     l2s   = diffs[:,0]**2 + diffs[:,1]**2
  #     # this is equivalent to the elementwise dot product
  #     dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
  #     t = np.clip(dots / l2s, 0.0, 1.0)
  #     projections = trajectory[:-1,:] + (t*diffs.T).T
  #     dists = np.linalg.norm(point - projections,axis=1)
  #     min_dist_segment = np.argmin(dists)
  #     return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

  def update_params(self, mu, h_cg, l_r, cs_f, cs_r, I_z, mass, exe_path):
    self.params = [mu, h_cg, l_r, cs_f, cs_r, I_z, mass]
    self.params_set = True
    if self.sim_p is None:
      # print('starting ex and setting map')
      self._start_executable(exe_path)
    self._set_map()
    # print('before creating proto')

    # create update proto
    update_param_proto = sim_requests_pb2.SimRequest()
    update_param_proto.type = 3
    update_param_proto.update_request.mu = mu
    update_param_proto.update_request.h_cg = h_cg
    update_param_proto.update_request.l_r = l_r
    update_param_proto.update_request.cs_f = cs_f
    update_param_proto.update_request.cs_r = cs_r
    update_param_proto.update_request.I_z = I_z
    update_param_proto.update_request.mass = mass
    # serialize reset proto
    update_param_string = update_param_proto.SerializeToString()
    # print('proto serialized')
    # send update param request
    self.socket.send(update_param_string)
    # print('Gym env - Update param request sent.')
    # receive response
    update_response_string = self.socket.recv()
    update_response_proto = sim_requests_pb2.SimResponse()
    update_response_proto.ParseFromString(update_response_string)
    if update_response_proto.update_resp.result:
      print('Gym env - Update param failed')
      return None

    # print('Gym env - params updated.')
    # start executable
    # self._start_executable()
    # call set map
    # self._set_map()

  def get_attr(self, attr_name, indices=None):
    raise NotImplementedError()

  def set_attr(self, attr_name, value, indices=None):
    raise NotImplementedError()

  def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
    raise NotImplementedError()

  def _get_target_envs(self, indices):
    raise NotImplementedError()

  def close(self):
    pass

  def seed(self):
    pass
