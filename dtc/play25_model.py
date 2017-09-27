
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


resolution = (3, 60, 80)
episodes_to_watch = 10
frame_repeat = 4

model_savefile = "./weights_long.dump"
# Configuration file path
config_file_path = '../ViZDoom/scenarios/defend_the_center.cfg'



def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    #game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()
    print("Doom initialized.")
    return game

def preprocess(image):
    image = transform.resize(image, resolution)
    image = np.ascontiguousarray(image, dtype=np.float32)
    image = torch.from_numpy(image)
    return image

def get_current_state():
    return preprocess(game.get_state().screen_buffer)

def get_q(input):
    input = Variable(input)
    return dqn(input)

def get_action(input):
    qs = get_q(input)
    val, ind = torch.max(qs, 1)
    return ind.data.numpy()[0]

game = initialize_vizdoom(config_file_path)
actions = [list(a) for a in it.product([0, 1], repeat=game.get_available_buttons_size())]
actions = actions[0:6]

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(1536, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, len(actions))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 1536)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

dqn = torch.load(model_savefile)
print(dqn)
for _ in range(episodes_to_watch):
    game.new_episode()
    while not game.is_episode_finished():
        state = get_current_state()
        state = state.view(1, 3, resolution[1], resolution[2])
        best_action_index = get_action(state)

        # Instead of make_action(a, frame_repeat) in order to make the animation smooth
        game.set_action(actions[best_action_index])
        for _ in range(frame_repeat):
            game.advance_action()

    # Sleep between episodes
    sleep(1.0)
    score = game.get_total_reward()
    print("Total score: ", score)

