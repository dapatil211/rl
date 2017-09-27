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
print("IMPORTS DONE")
env = RunEnv(visualize=False)
action_space = env.action_space
ob_space = env.observation_space
stepsize = tf.Variable(initial_value=np.float32(np.array(0.001)), name='stepsize')

with tf.variable_scope("global") as vs:
    with tf.variable_scope("pi"):
        policy = PolicyNet(ob_space.shape[0], action_space.shape[0], stepsize)
    with tf.variable_scope("vf"):
        value_func = ValueNet(ob_space.shape[0], action_space.shape[0])

print("CREATED NETWORKS")
global_counter = itertools.count()

num_threads = multiprocessing.cpu_count()
#num_threads = 1

workers = []

summary_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))

for i in range(num_threads):
    writer = None
    if i == 0:
        writer = summary_writer
    worker = Worker(
      name="worker_{}".format(i),
      env=RunEnv(visualize=False),
      policy_net=policy,
      value_net=value_func,
      global_counter=global_counter,
      summary_writer=writer,
      discount_factor = 0.99,
      max_global_steps=1e6,
      desired_kl=.002,
      stepsize=stepsize)
    workers.append(worker)

print("CREATED WORKERS")
saver = tf.train.Saver(keep_checkpoint_every_n_hours=1.0, max_to_keep=5)

def test(interval, sess, coord, policy):
    with sess.as_default(), sess.graph.as_default():
        while not coord.should_stop():
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
            step = next(global_counter)
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=total_reward, tag="eval/total_reward")
            episode_summary.value.add(simple_value=length, tag="eval/episode_length")
            summary_writer.add_summary(episode_summary, step)
            summary_writer.flush()
            
            saver.save(sess, CHECKPOINT_DIR + '/my-model', global_step=step)

            time.sleep(interval)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf_util.initialize()
    coord = tf.train.Coordinator()
    enqueue_threads = []
    for qr in [policy.q_runner, value_func.q_runner]:
        assert (qr != None)
        enqueue_threads.extend(qr.create_threads(sess, coord=coord, start=True))


    # Load a previous checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print("Loading model checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # Start worker threads
    worker_threads = []
    for worker in workers:
        worker_fn = lambda: worker.run(sess, coord)
        t = threading.Thread(target=worker_fn)
        t.start()
        worker_threads.append(t)

  # Start a thread for policy eval task
    monitor_thread = threading.Thread(target=lambda: test(60, sess, coord, policy))
    monitor_thread.start()

  # Wait for all workers to finish
    coord.join(worker_threads)
    env.close()

