
# coding: utf-8

# In[1]:

from vizdoom import *

import math
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import itertools as it
import pickle
from time import time, sleep

from collections import namedtuple
from copy import deepcopy
from PIL import Image
from skimage import transform, io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T


# In[2]:

# Q-learning settings
learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
epochs = 20
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 4
resolution = (1, 60, 80)
episodes_to_watch = 10

model_savefile = "./weights_pen.dump"
# Configuration file path
config_file_path = './defend_the_center_1.cfg'


# In[3]:

def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


# In[4]:

game = initialize_vizdoom(config_file_path)


# In[5]:

def preprocess(image):
    image = transform.resize(image, (resolution[1], resolution[2]))
    image = np.ascontiguousarray(image, dtype=np.float32)
    image = torch.from_numpy(image)
    return image

def get_current_state():
    return preprocess(game.get_state().screen_buffer), torch.FloatTensor(game.get_state().game_variables)


# In[6]:

actions = [list(a) for a in it.product([0, 1], repeat=game.get_available_buttons_size())]
actions = actions[0:6]

# In[7]:

class Replay:
    curIndex = 0
    size = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.replay = []
    def add(self, s1, s2, action, reward):
        if s2 is None:
           s2 = (torch.zeros(s1[0].size()), torch.zeros(s1[1].size()))
        if self.size == self.capacity:
            self.replay[self.curIndex] = (s1, s2, action, reward)
        else:
            self.replay.append((s1, s2, action, reward))
            self.size = min(self.size + 1, self.capacity)
        self.curIndex = (self.curIndex + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.replay, batch_size)
replay = Replay(replay_memory_size)


# In[8]:

replay.size


# In[9]:

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(1536 + 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, len(actions))

    def forward(self, x):
        vis, num = x
        x = F.relu(self.conv1(vis))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 1536)
        x = torch.cat([x, num], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
dqn = DQN()
#dqn = torch.load('sp05_weights.dump')


# In[10]:


mse = nn.MSELoss()
optimizer = optim.SGD(dqn.parameters(), lr=learning_rate)

def get_q(input):
    vis, num = input
    vis = Variable(vis)
    num = Variable(num)
    return dqn((vis, num))

def get_action(input):
    qs = get_q(input)
    val, ind = torch.max(qs, 1)
    return ind.data.numpy()[0]

def learn_replay():
    if batch_size <= replay.size:
        sample = zip(*replay.sample(batch_size))
        s1, s2, action, reward = sample

        vis1, num1 = zip(*s1)
        vis1 = torch.stack(vis1)
	vis1 = vis1.view(vis1.size()[0], 1, vis1.size()[1], vis1.size()[2])
        num1 = torch.stack(num1)
        #s1 = s1.view(s1.size()[0], 3, s1.size()[1], s1.size()[2])
        
        action = np.array(action)
        q1 = get_q((vis1, num1))
	q1 = q1.data.numpy()[np.arange(action.size), action]
       
        vis2, num2 = zip(*s2)
        vis2 = torch.stack(vis2)
	vis2 = vis2.view(vis2.size()[0], 1, vis2.size()[1], vis2.size()[2])
        num2 = torch.stack(num2)
        #s2 = torch.stack(s2)
        
        #s2 = s2.view(s2.size()[0], 3, s2.size()[1], s2.size()[2])
        q2 = get_q((vis2, num2)).data.numpy()
        q2 = np.max(q2, 1)
        #q_func = np.vectorize(lambda s:np.max(get_q(s.view(1, 1, resolution[0], resolution[1]))) if s else 0)
        #q2 = q_func(s2)
        
        y = reward + discount_factor * q2
        
        q1 = Variable(torch.from_numpy(q1), requires_grad=True)
        y = Variable(torch.from_numpy(y).float())
        loss = mse(q1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        


# In[11]:

def learn(eps):
    action = 0
    vis1, num1 = get_current_state()
    if random.random() < eps:
        action = random.randint(0, len(actions)-1)
    else:
        action = get_action((vis1.view(1, resolution[0], resolution[1], resolution[2]), num1.view(1, num1.shape[0])))
    reward = game.make_action(actions[action], frame_repeat)
    #if actions[action][2] == 1:
    #    reward -= .05
    s2 = None if game.is_episode_finished() else get_current_state()
    replay.add((vis1, num1), s2, action, reward)
    learn_replay()


# In[12]:

NUM_EPOCHS = 20
MAX_EPISODES=8000
EPS_START = .95
EPS_END = .1
EPS_CONST = MAX_EPISODES * .1
EPS_DECAY = MAX_EPISODES * .70
i = 0
global_training_steps = 0
num_episodes = 0
while num_episodes < MAX_EPISODES:
    print "%d EPOCH" % i
    i += 1
    learning_steps = 0
    epsilon = 0
    if num_episodes <= EPS_CONST:
        epsilon = EPS_START
    elif num_episodes <= EPS_DECAY:
        epsilon = EPS_START - (num_episodes - EPS_CONST) / (EPS_DECAY - EPS_CONST) * (EPS_START - EPS_END)
    else:
        epsilon = EPS_END
    scores = []
    while learning_steps < learning_steps_per_epoch:
        if(game.is_episode_finished() or num_episodes == 0):
            scores.append(game.get_total_reward())
            game.new_episode()
            num_episodes += 1
        learn(epsilon)
        learning_steps += 1
	global_training_steps += 1
    print("Epoch score: %d" % np.mean(scores))
    print("Training steps: %d" % global_training_steps)
    print("Episodes: %d" % num_episodes)
    torch.save(dqn, model_savefile)

game.close()
print("======================================")
print("Training finished. It's time to watch!")

# Reinitialize the game with window visible
game.set_window_visible(True)
game.set_mode(Mode.ASYNC_PLAYER)
game.init()

for _ in range(episodes_to_watch):
    game.new_episode()
    while not game.is_episode_finished():
        vis, num = get_current_state()
        best_action_index = get_action((vis.view(1, resolution[0], resolution[1], resolution[2]), num.view(1, num.shape[0])))

        # Instead of make_action(a, frame_repeat) in order to make the animation smooth
        game.set_action(actions[best_action_index])
        for _ in range(frame_repeat):
            game.advance_action()

    # Sleep between episodes
    sleep(1.0)
    score = game.get_total_reward()
    print("Total score: ", score)

