import tensorflow as tf
import numpy as np
from baselines import logger
from baselines.acktr.utils import kl_div
from baselines.common import tf_util
from baselines.acktr import kfac
from baselines import common



class PolicyNet(object):
    def __init__(self, ob_shape, num_actions, stepsize, scope="global"):

        self.state = tf.placeholder(shape=[None, ob_shape*2], dtype=tf.float32, name="x")
        error = tf.placeholder(shape=[None], name="error", dtype=tf.float32)
        action =  tf.placeholder(shape=[None, num_actions], name="action", dtype=tf.float32)
        fc1 = tf.contrib.layers.fully_connected(self.state, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        fc3 = tf.contrib.layers.fully_connected(fc2, 128)
        fc4 = tf.contrib.layers.fully_connected(fc3, 128)
        fc5 = tf.contrib.layers.fully_connected(fc4, 128)

        mu = tf.contrib.layers.fully_connected(fc5, num_actions, activation_fn=tf.sigmoid)
        sigma = tf.contrib.layers.fully_connected(fc5, num_actions, activation_fn=tf.nn.softplus)

        normal_dist = tf.contrib.distributions.Normal(mu, sigma)
        
        entropy = tf.reduce_sum(normal_dist.entropy(), 1)
        log_prob = tf.reduce_sum(normal_dist.log_prob(action), 1)

        #compute regular loss
        a_loss = - tf.reduce_mean((log_prob * error) + .01 * entropy)
        
        #compute fisher loss
        a_fisher_loss = -tf.reduce_mean(log_prob)

        #sample an action
        self.sampled_action = normal_dist.sample()
        
        #return distribution
        a_dist = tf.concat([tf.reshape(mu, [-1, num_actions]), tf.reshape(sigma, [-1, num_actions])], 1)

        self._act = tf_util.function([self.state], [self.sampled_action, a_dist])

        #kl-divergence
        a_old_dist = tf.placeholder(shape=a_dist.shape, dtype=tf.float32)

        kl = tf.reduce_mean(kl_div(a_old_dist, a_dist, num_actions))
        self.compute_kl = tf_util.function([self.state, a_old_dist], kl)
        
        tf.summary.scalar('policy_loss', a_loss)
        tf.summary.histogram('policy_entropy', entropy)
        tf.summary.scalar('policy_fisher_loss', a_fisher_loss)

        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries = [s for s in summary_ops if "pi" in s.name and "global" in s.name]
        summaries = tf.summary.merge(summaries)

        pi_var_list = []
        for var in tf.trainable_variables():
            if "pi" in var.name and scope in var.name:
                pi_var_list.append(var)
        optim = kfac.KfacOptimizer(learning_rate=stepsize, cold_lr=stepsize*(1-0.9), momentum=0.9, kfac_update=2,\
                                epsilon=1e-2, stats_decay=0.99, async=1, cold_iter=1, max_grad_norm=None)

        update_op, self.q_runner = optim.minimize(a_loss, a_fisher_loss, var_list=pi_var_list)
        self.do_update = tf_util.function([self.state, error, action], [update_op, summaries]) #pylint: disable=E1101

    
    def act(self, ob):
        ac, ac_dist = self._act(ob[None])
        return ac[0], ac_dist[0]

class ValueNet(object):
    def __init__(self, ob_shape, num_actions, scope="global"):
        print(ob_shape)
        print(num_actions)
        state = tf.placeholder(shape=[None, ob_shape * 2 + num_actions + 1], dtype=tf.float32, name="x")
        target = tf.placeholder(shape=[None], name="target", dtype=tf.float32)

        fc1 = tf.contrib.layers.fully_connected(state, 256)
        fc2 = tf.contrib.layers.fully_connected(fc1, 256)
        fc3 = tf.contrib.layers.fully_connected(fc2, 128)
        fc4 = tf.contrib.layers.fully_connected(fc3, 128)
        fc5 = tf.contrib.layers.fully_connected(fc4, 128)

        value = tf.contrib.layers.fully_connected(fc5, 1)
        sample_value = value + tf.random_normal(tf.shape(value))

        # Compute value
        self._predict = tf_util.function([state], value)

        loss = tf.reduce_mean(tf.square(value - target))
        sample_loss = tf.reduce_mean(tf.square(value - tf.stop_gradient(sample_value)))

        tf.summary.scalar('vf_loss', loss)
        tf.summary.scalar('vf_fisher_loss', sample_loss)
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries = [s for s in summary_ops if "vf" in s.name and "global" in s.name]
        summaries = tf.summary.merge(summaries)
        
        vf_var_list = []
        for var in tf.trainable_variables():
            if "vf" in var.name and scope in var.name:
                vf_var_list.append(var)

        optim = kfac.KfacOptimizer(learning_rate=0.001, cold_lr=0.001*(1-0.9), momentum=0.9, \
                                    clip_kl=0.3, epsilon=0.1, stats_decay=0.95, \
                                    async=1, kfac_update=2, cold_iter=50, \
                                    max_grad_norm=None)
        update_op, self.q_runner = optim.minimize(loss, sample_loss, var_list=vf_var_list)
        self.do_update = tf_util.function([state, target], update_op) #pylint: disable=E1101
        self.summaries = tf_util.function([state, target], summaries) #pylint: disable=E1101

    def _preproc(self, states, actions):
        X = np.stack([np.concatenate([s, a, np.ones(1)], -1) for (s, a) in zip(states, actions)])
        return X
        #l = pathlength(path)
        #al = np.arange(l).reshape(-1,1)/10.0
        #act = path["action_dist"].astype('float32')
        #X = np.concatenate([path['observation'], act, al, np.ones((l, 1))], axis=1)
        #return X

    def predict(self, state, action):
        return self._predict(np.concatenate([state, action, np.ones(1)])[np.newaxis, ...])
    
    def fit(self, states, actions, values):
        X = self._preproc(states, actions)
        #logger.record_tabular("EVBefore", common.explained_variance(self._predict(X), values))
        for _ in range(25): self.do_update(X, values)
        #logger.record_tabular("EVAfter", common.explained_variance(self._predict(X), values))
        return self.summaries(X, values)
