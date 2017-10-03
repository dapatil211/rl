from model import PolicyNet, ValueNet
from worker import Worker
import unittest
import time
import gym
import sys
import os
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing
from baselines.common import tf_util

from osim.env import RunEnv

MODEL_DIR = "model/"
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
env = RunEnv(visualize=True)
action_space = env.action_space
ob_space = env.observation_space
stepsize = tf.Variable(initial_value=np.float32(np.array(0.001)), name='stepsize')

with tf.variable_scope("global") as vs:
    with tf.variable_scope("pi"):
        policy = PolicyNet(ob_space.shape[0], action_space.shape[0], stepsize)
    with tf.variable_scope("vf"):
        value_func = ValueNet(ob_space.shape[0], action_space.shape[0])

print("CREATED NETWORKS")
saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0, max_to_keep=5)

def test(interval, sess, coord, policy):
    with sess.as_default(), sess.graph.as_default():
        for i in range(10):
            ob = np.array(env.reset())
            prev_ob = np.float32(np.zeros(ob.shape))
            state = np.concatenate([ob, prev_ob], -1)
            done = False
            total_reward = 0
            length = 0
            while not done:
                state = state[np.newaxis, ...]
                feed = { policy.state: state }
                action = sess.run([policy.sampled_action], feed_dict=feed)
                prev_ob = np.copy(ob)
                ob, reward, done, _ = env.step(action[0][0])
                ob = np.array(ob)
                state = np.concatenate([ob, prev_ob], -1)
                total_reward += reward
                length += 1
            time.sleep(2)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf_util.initialize()
    coord = tf.train.Coordinator()

    # Load a previous checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print("Loading model checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)


    test(300, sess, coord, policy)

  # Wait for all workers to finish
    env.close()

