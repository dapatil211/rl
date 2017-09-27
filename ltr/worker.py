import tensorflow as tf
import numpy as np
from baselines.common import tf_util
from model import PolicyNet, ValueNet
import collections



def make_copy_params_op(v1_list, v2_list):
  """
  Creates an operation that copies parameters from variable in v1_list to variables in v2_list.
  The ordering of the variables in the lists must be identical.
  """
  v1_list = list(sorted(v1_list, key=lambda v: v.name))
  v2_list = list(sorted(v2_list, key=lambda v: v.name))

  update_ops = []
  for v1, v2 in zip(v1_list, v2_list):
    op = v2.assign(v1)
    update_ops.append(op)

  return update_ops

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "action_dist", "done"])

def rollout(env, policy, counter):
    path = []
    ob = np.array(env.reset())
    prev_ob = np.float32(np.zeros(ob.shape))
    state = np.concatenate([ob, prev_ob], -1)
    for i in range(env.spec.timestep_limit):
        global_count = next(counter)
        if global_count % 100 == 0:
            print(global_count)
        action, action_dist = policy.act(state)
        prev_ob = np.copy(ob)
        ob, reward, done, _ = env.step(action)
        ob = np.array(ob)
        next_state = np.concatenate([ob, prev_ob], -1)

        path.append(Transition(state, action, reward, next_state, action_dist, done))
        state = next_state
        if done:
            break
    return path, global_count

class Worker():
    def __init__(self, name, policy_net, value_net, env, global_counter, stepsize, discount_factor=0.99, summary_writer=None, max_global_steps=None, desired_kl=.002):
        self.name = name
        self.discount_factor = discount_factor
        self.max_global_steps = max_global_steps
        self.global_policy_net = policy_net
        self.global_value_net = value_net
        self.global_counter = global_counter
        self.summary_writer = summary_writer
        self.env = env
        self.desired_kl = desired_kl
        self.stepsize = stepsize

        with tf.variable_scope(name):
            with tf.variable_scope("pi"):
                self.policy_net = PolicyNet(env.observation_space.shape[0], env.action_space.shape[0], stepsize, name)
            with tf.variable_scope("vf"):
                self.value_net = ValueNet(env.observation_space.shape[0], env.action_space.shape[0], name)

        # copy over the global network to the local training network
        self.copy_params_op = make_copy_params_op(
              tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
              tf.contrib.slim.get_variables(scope=self.name, collection=tf.GraphKeys.TRAINABLE_VARIABLES))

        self.state = None


    def run(self, sess, coord, timesteps_per_batch=100):
        with sess.as_default(), sess.graph.as_default():
            global_count = 0
            while not coord.should_stop():
                sess.run(self.copy_params_op)
                timesteps = 0
                paths = []
                while timesteps < timesteps_per_batch:
                    path, global_count = rollout(self.env, self.policy_net, self.global_counter)
                    timesteps += len(path)
                    print("%s: %d" % (self.name, timesteps))
                    paths.append(path)

                states = []
                errors = []
                values = []
                actions = []
                action_dists = []
                for path in paths:
                    reward = 0.
                    if not path[-1].done:
                        reward = self.value_net.predict(path[-1].state, path[-1].action)

                    for transition in path:
                        reward = transition.reward + self.discount_factor * reward
                        states.append(transition.state)
                        errors.append(reward - self.value_net.predict(transition.state, transition.action))
                        values.append(reward)
                        action_dists.append(transition.action_dist)
                        actions.append(transition.action)

                states = np.array(states)
                errors = np.array(errors).squeeze()
                values = np.array(values)
                actions = np.array(actions)
                action_dists = np.array(action_dists)

                v_summary = self.global_value_net.fit(states, actions, values)

                _, p_summary = self.global_policy_net.do_update(states, errors, actions)
                kl = self.global_policy_net.compute_kl(states, action_dists)
                
                if kl > self.desired_kl * 2:
                    tf_util.eval(tf.assign(self.stepsize, self.stepsize / 1.5))
                elif kl < self.desired_kl * .5:
                    tf_util.eval(tf.assign(self.stepsize, self.stepsize * 1.5))
                if self.summary_writer:
                    self.summary_writer.add_summary(p_summary, global_count)
                    self.summary_writer.add_summary(v_summary, global_count)
                    self.summary_writer.flush()
                
                if global_count > self.max_global_steps:
                    coord.request_stop()


