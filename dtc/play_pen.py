
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
from skimage import transform, io, color
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T


resolution = (1, 60, 80)
episodes_to_watch = 10
frame_repeat = 4

model_savefile = "./weights_pen.dump"
# Configuration file path
config_file_path = './defend_the_center.cfg'



def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    #game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()
    print("Doom initialized.")
    return game

def preprocess(image):
    print image.shape
    image = transform.resize(image, (resolution[1], resolution[2]))
    image = np.ascontiguousarray(image, dtype=np.float32)
    image = torch.from_numpy(image)
    return image

def get_current_state():
    return preprocess(game.get_state().screen_buffer), torch.FloatTensor(game.get_state().game_variables)

def get_q(input):
    vis, num = input
    vis = Variable(vis)
    num = Variable(num)
    return dqn((vis, num))

def get_action(input):
    qs = get_q(input)
    val, ind = torch.max(qs, 1)
    return ind.data.numpy()[0]

game = initialize_vizdoom(config_file_path)
actions = [list(a) for a in it.product([0, 1], repeat=game.get_available_buttons_size())]

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
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
#dqn = DQN()
dqn = torch.load(model_savefile)

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

