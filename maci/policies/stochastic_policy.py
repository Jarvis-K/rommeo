
from maci.environments.env_spec import MAEnvSpec

import tensorflow as tf

from maci.core.serializable import Serializable

from maci.misc.nn import feedforward_net

from .nn_policy import NNPolicy
import numpy as np

class StochasticNNPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy."""

    def __init__(self,
                 env_spec=None,
                 observation_space=None,
                 action_space=None,
                 hidden_layer_sizes=(100, 100),
                 squash=False,
                 squash_func=tf.tanh,
                 name='stochastic_policy',
                 u_range=1.,
                 shift=None,
                 scale=None,
                 joint=False, agent_id=None, sampling=False):
        Serializable.quick_init(self, locals())
        if env_spec is None:
            self._observation_dim = observation_space.flat_dim
            self._action_dim = action_space.flat_dim
        elif isinstance(env_spec, MAEnvSpec):
            assert agent_id is not None
            self._observation_dim = env_spec.observation_space[agent_id].flat_dim
            if joint:
                self._action_dim = env_spec.action_space.flat_dim
            else:
                self._action_dim = env_spec.action_space[agent_id].flat_dim
        else:
            self._action_dim = env_spec.action_space.flat_dim
            self._observation_dim = env_spec.observation_space.flat_dim
        self._layer_sizes = list(hidden_layer_sizes) + [self._action_dim]
        self._squash = squash
        self._squash_func = squash_func
        self._u_range = u_range
        self.shift = shift
        self.scale = scale
        self._name = name + '_agent_{}'.format(agent_id)
        self.sampling = sampling
        self.agent_id = agent_id

        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation_{}_agent_{}'.format(name, agent_id))
        # self._observation_ph = None
        # self._actions = None


        self._actions = self.actions_for(self._observation_ph)

        super(StochasticNNPolicy, self).__init__(
            env_spec, self._observation_ph, self._actions, self._name)

    def actions_for(self, observations, n_action_samples=1, reuse=False):

        n_state_samples = tf.shape(observations)[0]

        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (n_state_samples, n_action_samples,
                            self._action_dim)
        else:
            latent_shape = (n_state_samples, self._action_dim)

        latents = tf.random_normal(latent_shape)

        with tf.variable_scope(self._name, reuse=reuse):
            raw_actions = feedforward_net(
                (observations, latents),
                layer_sizes=self._layer_sizes,
                activation_fn=tf.nn.relu,
                output_nonlinearity=None)

        if self.sampling:
            u = tf.random_uniform(tf.shape(raw_actions))
            return tf.nn.softmax(raw_actions - tf.log(-tf.log(u)), axis=-1)

        if (self.shift is not None) and (self.scale is not None) and self._squash:
            tf.scalar_mul(self.scale, tf.tanh(raw_actions) + self.shift)
        print('stochastic', self._u_range, self._squash, self._squash_func)
        return tf.scalar_mul(self._u_range, self._squash_func(raw_actions)) if self._squash else tf.clip_by_value(raw_actions, -self._u_range, self._u_range)



class StochasticNNConditionalPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy."""

    def __init__(self,
                 env_spec=None,
                 observation_space=None,
                 action_space=None,
                 opponent_action_space=None,
                 hidden_layer_sizes=(100, 100),
                 squash=False,

                 squash_func=tf.tanh,
                 name='conditional_policy',
                 u_range=1.,
                 shift=None,
                 scale=None,
                 joint=False, agent_id=None, sampling=False):
        Serializable.quick_init(self, locals())
        self.agent_id = agent_id
        if env_spec is None:
            self._observation_dim = observation_space.flat_dim
            self._action_dim = action_space.flat_dim
            self._opponent_action_dim = opponent_action_space.flat_dim
        else:
            assert isinstance(env_spec, MAEnvSpec)
            assert agent_id is not None
            self._action_dim = env_spec.action_space[agent_id].flat_dim
            self._observation_dim = env_spec.observation_space[agent_id].flat_dim
            self._opponent_action_dim = env_spec.action_space.opponent_flat_dim(agent_id)
        print('opp dim', self._opponent_action_dim)
        self._layer_sizes = list(hidden_layer_sizes) + [self._opponent_action_dim]
        self._squash = squash
        self._squash_func = squash_func
        self._u_range = u_range
        self.shift = shift
        self.scale = scale
        self.sampling = sampling
        self._name = name + '_agent_{}'.format(agent_id)

        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation_{}_agent_{}'.format(name, agent_id))
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._action_dim],
            name='actions_{}_agent_{}'.format(name, agent_id))
        self._opponent_actions = self.actions_for(self._observation_ph, self._actions_ph)

        super(StochasticNNConditionalPolicy, self).__init__(
            env_spec, self._observation_ph, self._opponent_actions, self._name)

    def get_action(self, observation, action):
        return self.get_actions(observation[None], action[None])[0], None

    def get_actions(self, observations, self_actions):
        feeds = {self._observation_ph: observations, self._actions_ph: self_actions}
        actions = tf.get_default_session().run(self._opponent_actions, feeds)
        return actions

    def actions_for(self, observations, actions, n_action_samples=1, reuse=False):

        n_state_samples = tf.shape(observations)[0]
        # n_action_samples = tf.shape(actions)[0]
        # assert n_state_samples == n_action_samples

        if n_action_samples > 1:
            observations = observations[:, None, :]
            actions = actions[:, None, :]
            latent_shape = (n_state_samples, n_action_samples,
                            self._opponent_action_dim)
        else:
            latent_shape = (n_state_samples, self._opponent_action_dim)

        latents = tf.random_normal(latent_shape)
        # print('latents', latents)
        with tf.variable_scope(self._name, reuse=reuse):
            raw_actions = feedforward_net(
                (observations, actions, latents),
                layer_sizes=self._layer_sizes,
                activation_fn=tf.nn.relu,
                output_nonlinearity=None)

        if self.sampling:
            # print('raw_actions', raw_actions)
            u = tf.random_uniform(tf.shape(raw_actions))
            return tf.nn.softmax(raw_actions - tf.log(-tf.log(u)), axis=-1)

        if (self.shift is not None) and (self.scale is not None) and self._squash:
            tf.scalar_mul(self.scale, tf.tanh(raw_actions) + self.shift)
        print('cond stochastic', self._u_range, self._squash, self._squash_func)
        return tf.scalar_mul(self._u_range, self._squash_func(raw_actions)) if self._squash else tf.clip_by_value(raw_actions,
                                                                                                        -self._u_range,
                                                                                                        self._u_range)


class MarginalPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy."""

    def __init__(self,
                 env_spec=None,
                 observation_space=None,
                 action_space=None,
                 hidden_layer_sizes=(100, 100),
                 squash=False,
                 squash_func=tf.tanh,
                 name='stochastic_policy',
                 u_range=1.,
                 shift=None,
                 scale=None,
                 joint=False, agent_id=None, sampling=False):
        Serializable.quick_init(self, locals())
        if env_spec is None:
            self._observation_dim = observation_space.flat_dim
            self._action_dim = action_space.flat_dim
        elif isinstance(env_spec, MAEnvSpec):
            assert agent_id is not None
            self._observation_dim = env_spec.observation_space[agent_id].flat_dim
            self._action_dim = env_spec.action_space[agent_id].flat_dim
        else:
            self._action_dim = env_spec.action_space.flat_dim
            self._observation_dim = env_spec.observation_space.flat_dim
        self._layer_sizes = list(hidden_layer_sizes) + [1]
        self._squash = squash
        self._squash_func = squash_func
        self._u_range = u_range
        self.shift = shift
        self.scale = scale
        self._name = name + '_agent_{}'.format(agent_id)
        self.sampling = sampling
        self.agent_id = agent_id

        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observations'.format(name, agent_id))
        # self._observation_ph = None
        # self._actions = None


        self._actions = self.actions_for(self._observation_ph)

        super(MarginalPolicy, self).__init__(
            env_spec, self._observation_ph, self._actions, self._name)


    def get_actions(self, observations):
        feeds = {self._observation_ph: observations}
        actions = tf.get_default_session().run(self._actions, feeds)
        return actions

    def actions_for(self, observations, n_action_samples=1, reuse=False):
        observations = tf.reshape(observations, (-1, self._observation_dim))
        n_state_samples = tf.shape(observations)[0]

        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (n_state_samples, n_action_samples,
                            self._action_dim)
        else:
            latent_shape = (n_state_samples, self._action_dim)

        latents = tf.random_normal(latent_shape)

        with tf.variable_scope(self._name, reuse=reuse):
            raw_actions = feedforward_net(
                (observations, latents),
                layer_sizes=self._layer_sizes,
                activation_fn=tf.nn.relu,
                output_nonlinearity=None)

        if self.sampling:
            u = tf.random_uniform(tf.shape(raw_actions))
            return tf.nn.softmax(raw_actions - tf.log(-tf.log(u)), axis=-1)

        if (self.shift is not None) and (self.scale is not None) and self._squash:
            tf.scalar_mul(self.scale, tf.tanh(raw_actions) + self.shift)
        print('stochastic', self._u_range, self._squash, self._squash_func)
        return tf.scalar_mul(self._u_range, self._squash_func(raw_actions)) if self._squash else tf.clip_by_value(raw_actions, -self._u_range, self._u_range)

class ConditionedStochasticNNPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy."""

    def __init__(self,
                 env_spec=None,
                 observation_space=None,
                 action_space=None,
                 opponent_action_space=None,
                 hidden_layer_sizes=(100, 100),
                 squash=False,
                 squash_func=tf.tanh,
                 name='conditioned_policy',
                 u_range=1.,
                 shift=None,
                 scale=None,
                 joint=False, agent_id=None, sampling=False):
        Serializable.quick_init(self, locals())
        self.agent_id = agent_id
        if env_spec is None:
            self._observation_dim = observation_space.flat_dim
            self._action_dim = action_space.flat_dim
            # self._oppo_observation_dim = opponent_space.flat
            self._oppo_action_dim = opponent_action_space.flat_dim
        else:
            assert isinstance(env_spec, MAEnvSpec)
            assert agent_id is not None
            self._action_dim = 1
            self._observation_dim = env_spec.observation_space[agent_id].flat_dim
            self._oppo_observation_dim = self._observation_dim * (self._observation_dim - 1)
            self._oppo_action_dim = self._action_dim * (self._observation_dim - 1)
        print('opp dim', self._oppo_action_dim)
        self._my_layer_sizes = list(hidden_layer_sizes)
        self._oppo_layer_sizes = list(hidden_layer_sizes)
        self._output_layer_sizes = [2* hidden_layer_sizes[-1], 100, self._action_dim]
        self._squash = squash
        self._squash_func = squash_func
        self._u_range = u_range
        self.shift = shift
        self.scale = scale
        self.sampling = sampling
        self._name = name + '_agent_{}'.format(agent_id)

        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation_{}_agent_{}'.format(name, agent_id))
        self._oppo_observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._oppo_observation_dim],
            name='observation_{}_oppo_{}'.format(name, agent_id))
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._action_dim],
            name='actions_{}_agent_{}'.format(name, agent_id))

        self._oppo_actions_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._oppo_action_dim],
            name='actions_{}_oppo_{}'.format(name, agent_id))

        self._actions, _ = self.actions_for(self._observation_ph, self._actions_ph)

        super(ConditionedStochasticNNPolicy, self).__init__(
            env_spec, self._observation_ph, self._actions, self._name)

    def get_joint_actions(self, observations, actions, n_action_samples=1):
        feeds = {self._observation_ph: observations, self._actions_ph: actions}
        observations = tf.reshape(observations, (-1, self._observation_dim))
        action, oppo_action = tf.get_default_session().run(self.actions_for(observations, actions, n_action_samples))
        joint_actions = np.zeros((action.shape[0], action.shape[1], action.shape[-1]+ oppo_action.shape[-1]))
        return tf.concat([action, oppo_action], aixs = -1)


    def get_actions(self, observations, actions):
        feeds = {self._observation_ph: observations, self._actions_ph: actions}
        # import pdb; pdb.set_trace()
        # oppos = tf.get_default_session().run(self.get_oppos(self._observation_ph, self._actions_ph), feeds)
        actions = tf.get_default_session().run(self._actions, feeds)
        return actions


    # def get_oppos(self, observations, actions):
    #     n_state_samples = tf.shape(observations)[0]
    #     # n_action_samples = tf.shape(actions)[0]
    #     # assert n_state_samples == n_action_samples
    #     actions = tf.reshape(actions, (-1, self._action_dim))
    #     oppo_observations = tf.reshape(observations, (-1, self._observation_dim, self._observation_dim))
    #     oppo_observations = tf.repeat(observations, self._observation_dim, axis=0)
    #     np_tmpl = np.array([i for i in range(self._observation_dim)])
    #     drop_indexs = tf.convert_to_tensor(np.array([np.delete(np_tmpl, i) for i in range(self._observation_dim)]))
    #     drop_indexs = tf.tile(drop_indexs, [n_state_samples, 1])
    #     oppo_observations = tf.gather(oppo_observations, drop_indexs, axis = 1, batch_dims=-1)
    #     oppo_actions = tf.reshape(actions, (-1, self._observation_dim, self._action_dim))
    #     oppo_actions = tf.repeat(oppo_actions, self._observation_dim, axis = 0)
    #     # oppo_actions = tf.gather(oppo_actions, drop_indexs, axis = 1, batch_dims=-1)

    #     oppo_observations = tf.reshape(oppo_observations,(-1, self._oppo_observation_dim))
    #     # oppo_actions = tf.reshape(oppo_actions, (-1, self._oppo_action_dim))

    #     return oppo_observations, oppo_actions, drop_indexs

    def actions_for(self, observations, actions, n_action_samples=1, reuse=False):
        observations = tf.reshape(observations, (-1,self._observation_dim))
        n_state_samples = tf.shape(observations)[0]
        # n_action_samples = tf.shape(actions)[0]
        # assert n_state_samples == n_action_samples
        actions = tf.reshape(actions, (-1, self._action_dim))
        oppo_observations = tf.reshape(observations, (-1, self._observation_dim, self._observation_dim))
        oppo_observations = tf.repeat(oppo_observations, self._observation_dim, axis=0)
        np_tmpl = np.array([i for i in range(self._observation_dim)])
        drop_indexs = tf.convert_to_tensor(np.array([np.delete(np_tmpl, i) for i in range(self._observation_dim)]))
        drop_indexs = tf.tile(drop_indexs, [n_state_samples//self._observation_dim, 1])
        oppo_observations = tf.gather(oppo_observations, drop_indexs, axis = 1, batch_dims=-1)
        oppo_actions = tf.reshape(actions, (-1, self._observation_dim, self._action_dim))
        joint_actions = tf.repeat(oppo_actions, self._observation_dim, axis = 0)
        oppo_actions = tf.gather(joint_actions, drop_indexs, axis = 1, batch_dims=-1)
        oppo_observations = tf.reshape(oppo_observations, (-1, self._observation_dim * (self._observation_dim -1)))
        oppo_actions = tf.reshape(oppo_actions, (-1, self._action_dim*(self._observation_dim -1)))
        # import pdb; pdb.set_trace()

        if n_action_samples > 1:
            observations = observations[:, None, :]
            observations = tf.repeat(observations, n_action_samples, axis=1)
            oppo_observations = oppo_observations[:, None, :]
            oppo_observations = tf.repeat(oppo_observations, n_action_samples, axis=1)
            actions = actions[:, None, :]
            actions = tf.repeat(actions, n_action_samples, axis=1)
            oppo_actions = oppo_actions[:, None, :]
            oppo_actions = tf.repeat(oppo_actions, n_action_samples, axis=1)
            joint_actions = joint_actions[:, None, :]
            joint_actions = tf.repeat(joint_actions, n_action_samples, axis=1)
            latent_shape = (n_state_samples, n_action_samples,
                            self._action_dim)
        else:
            latent_shape = (n_state_samples, self._action_dim)

        latents = tf.random_normal(latent_shape)
        # print('latents', latents)
        with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
            # import pdb; pdb.set_trace()
            h_act = tf.concat([observations, actions], axis = -1)
            h_act = tf.layers.dense(h_act, units=100, activation=tf.nn.relu)
            h_act = tf.layers.dense(h_act, units=100, activation=tf.nn.relu)
            h_oppo = tf.concat([oppo_observations, oppo_actions], axis = -1)
            h_oppo = tf.layers.dense(h_oppo, units=100, activation=tf.nn.relu)
            h_oppo = tf.layers.dense(h_oppo, units=100, activation=tf.nn.relu)
            raw_actions = tf.concat([h_act, h_oppo, latents], axis=-1)
            # raw_actions = tf.concat([h_act, latents], axis=-1)
            raw_actions = tf.layers.dense(raw_actions, units=100, activation=tf.nn.relu)
            raw_actions = tf.layers.dense(raw_actions, units=self._action_dim, activation=tf.nn.relu)

            # self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
            self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
    

        if self.sampling:
            # print('raw_actions', raw_actions)
            u = tf.random_uniform(tf.shape(raw_actions))
            return tf.nn.softmax(raw_actions - tf.log(-tf.log(u)), axis=-1)

        if (self.shift is not None) and (self.scale is not None) and self._squash:
            tf.scalar_mul(self.scale, tf.tanh(raw_actions) + self.shift)
        print('cond stochastic', self._u_range, self._squash, self._squash_func)
        return tf.scalar_mul(self._u_range, self._squash_func(raw_actions)) if self._squash else tf.clip_by_value(raw_actions,
                                                                                                        -self._u_range,
                                                                                                        self._u_range), oppo_actions

class JStochasticNNPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy."""

    def __init__(self,
                 env_spec=None,
                 observation_space=None,
                 action_space=None,
                 hidden_layer_sizes=(100, 100),
                 squash=False,
                 squash_func=tf.tanh,
                 name='jstochastic_policy',
                 u_range=1.,
                 shift=None,
                 scale=None,
                 joint=False, agent_id=None, sampling=False):
        Serializable.quick_init(self, locals())
        if env_spec is None:
            self._observation_dim = observation_space
            self._action_dim = action_space
        elif isinstance(env_spec, MAEnvSpec):
            assert agent_id is not None
            self._observation_dim = env_spec.observation_space[agent_id].flat_dim **2
            if joint:
                self._action_dim = env_spec.action_space.flat_dim
            else:
                self._action_dim = env_spec.action_space[agent_id].flat_dim
        else:
            self._action_dim = env_spec.action_space.flat_dim
            self._observation_dim = env_spec.observation_space.flat_dim **2
        self._layer_sizes = list(hidden_layer_sizes) + [self._action_dim]
        self._squash = squash
        self._squash_func = squash_func
        self._u_range = u_range
        self.shift = shift
        self.scale = scale
        self._name = name + '_agent_{}'.format(agent_id)
        self.sampling = sampling
        self.agent_id = agent_id

        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation_{}_agent_{}'.format(name, agent_id))
        # self._observation_ph = None
        # self._actions = None


        self._actions = self.actions_for(self._observation_ph)

        super(JStochasticNNPolicy, self).__init__(
            env_spec, self._observation_ph, self._actions, self._name)


    def get_actions(self, observations):
        feeds = {self._observation_ph: observations.reshape(-1, self._observation_dim)}
        actions = tf.get_default_session().run(self._actions, feeds)
        return actions

    def actions_for(self, observations, n_action_samples=1, reuse=False):
        n_state_samples = tf.shape(observations)[0]

        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (n_state_samples, n_action_samples,
                            self._action_dim)
        else:
            latent_shape = (n_state_samples, self._action_dim)

        latents = tf.random_normal(latent_shape)

        with tf.variable_scope(self._name, reuse=reuse):
            raw_actions = feedforward_net(
                (observations, latents),
                layer_sizes=self._layer_sizes,
                activation_fn=tf.nn.relu,
                output_nonlinearity=None)

        if self.sampling:
            u = tf.random_uniform(tf.shape(raw_actions))
            return tf.nn.softmax(raw_actions - tf.log(-tf.log(u)), axis=-1)

        if (self.shift is not None) and (self.scale is not None) and self._squash:
            tf.scalar_mul(self.scale, tf.tanh(raw_actions) + self.shift)
        print('stochastic', self._u_range, self._squash, self._squash_func)
        return tf.scalar_mul(self._u_range, self._squash_func(raw_actions)) if self._squash else tf.clip_by_value(raw_actions, -self._u_range, self._u_range)

class SSConditionedStochasticNNPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy."""

    def __init__(self,
                 env_spec=None,
                 observation_space=None,
                 action_space=None,
                 hidden_layer_sizes=(100, 100),
                 squash=False,
                 squash_func=tf.tanh,
                 name='sscstochastic_policy',
                 u_range=1.,
                 shift=None,
                 scale=None,
                 joint=False, agent_id=None, sampling=False):
        Serializable.quick_init(self, locals())
        if env_spec is None:
            self._observation_dim = observation_space
            self._action_dim = action_space
        elif isinstance(env_spec, MAEnvSpec):
            assert agent_id is not None
            self._observation_dim = env_spec.observation_space.flat_dim
            self._action_dim = env_spec.action_space.flat_dim
        else:
            self._action_dim = env_spec.action_space.flat_dim
            self._observation_dim = env_spec.observation_space.flat_dim **2
        self._layer_sizes = list(hidden_layer_sizes) + [1]
        self._squash = squash
        self._squash_func = squash_func
        self._u_range = u_range
        self.shift = shift
        self.scale = scale
        self._name = name + '_agent_{}'.format(agent_id)
        self.sampling = sampling
        self.agent_id = agent_id

        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation_{}_agent_{}'.format(name, agent_id))
        # self._actions_ph = tf.placeholder(
        #     tf.float32,
        #     shape=[None, self._observation_dim],
        #     name='actions_{}_agent_{}'.format(name, agent_id))
        # self._observation_ph = None
        # self._actions = None

        self._actions = self.actions_for(self._observation_ph)
        
        super(SSConditionedStochasticNNPolicy, self).__init__(
            env_spec, self._observation_ph, self._actions, self._name)


    def get_actions(self, observations):
        feeds = {self._observation_ph: observations.reshape(-1, self._observation_dim)}
        actions = tf.get_default_session().run(self._actions, feeds)
        return actions

    def get_params_internal(self,):
        scope = self._scope_name
            # Add "/" to 'scope' unless it's empty (otherwise get_collection will
            # return all parameters that start with 'scope'.

        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    def actions_for(self, observations, n_action_samples=1, reuse=False):
        n_state_samples = tf.shape(observations)[0]
        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (n_state_samples, n_action_samples,
                            1)
            
        else:
            latent_shape = (n_state_samples, 1)

        latents = tf.random_normal(latent_shape)
        my_actions = []
        for i in range(3):
            with tf.variable_scope(self._name+str(i), reuse=reuse):
                my_obs_dim = self._observation_dim//3 *(i+1)
                my_action_dim = self._action_dim//3 *(i+1)
                if n_action_samples > 1:
                    my_obs = observations[:, :, :my_obs_dim]
                else:
                    my_obs = observations[:, :my_obs_dim]
                if i > 0:
                    my_action_pl = tf.concat(my_actions, axis=-1)
                    my_actions.append(feedforward_net(
                        (my_obs, my_action_pl, latents),
                        layer_sizes=self._layer_sizes,
                        activation_fn=tf.nn.relu,
                        output_nonlinearity=None, name = self._name+str(i))
                    )
                    
                else:
                    my_actions.append(feedforward_net(
                        (my_obs, latents),
                        layer_sizes=self._layer_sizes,
                        activation_fn=tf.nn.relu,
                        output_nonlinearity=None, name=self._name+str(i))
                    )
        raw_actions = tf.concat(my_actions, -1)

        if self.sampling:
            u = tf.random_uniform(tf.shape(raw_actions))
            return tf.nn.softmax(raw_actions - tf.log(-tf.log(u)), axis=-1)

        if (self.shift is not None) and (self.scale is not None) and self._squash:
            tf.scalar_mul(self.scale, tf.tanh(raw_actions) + self.shift)
        print('stochastic', self._u_range, self._squash, self._squash_func)
        return tf.scalar_mul(self._u_range, self._squash_func(raw_actions)) if self._squash else tf.clip_by_value(raw_actions, -self._u_range, self._u_range)
        



class CConditionedStochasticNNPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy."""

    def __init__(self,
                 env_spec=None,
                 observation_space=None,
                 action_space=None,
                 nego_round = 1,
                 hidden_layer_sizes=(100, 100),
                 squash=False,
                 squash_func=tf.tanh,
                 name='ccstochastic_policy',
                 u_range=1.,
                 shift=None,
                 scale=None,
                 joint=False, agent_id=None, sampling=False):
        Serializable.quick_init(self, locals())
        if env_spec is None:
            self._observation_dim = observation_space
            self._action_dim = action_space
        elif isinstance(env_spec, MAEnvSpec):
            assert agent_id is not None
            self._observation_dim = env_spec.observation_space.flat_dim
            self._action_dim = env_spec.action_space.flat_dim
        else:
            self._action_dim = env_spec.action_space.flat_dim
            self._observation_dim = env_spec.observation_space.flat_dim **2
        self._layer_sizes = list(hidden_layer_sizes) + [1]
        self._squash = squash
        self._squash_func = squash_func
        self._u_range = u_range
        self.shift = shift
        self.scale = scale
        self._name = name + '_agent_{}'.format(agent_id)
        self.sampling = sampling
        self.agent_id = agent_id
        self.nego_round = nego_round

        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation_{}_agent_{}'.format(name, agent_id))
        # self._actions_ph = tf.placeholder(
        #     tf.float32,
        #     shape=[None, self._observation_dim],
        #     name='actions_{}_agent_{}'.format(name, agent_id))
        # self._observation_ph = None
        # self._actions = None

        self._actions = self.actions_for(self._observation_ph)
        
        super(CConditionedStochasticNNPolicy, self).__init__(
            env_spec, self._observation_ph, self._actions, self._name)


    def get_actions(self, observations):
        feeds = {self._observation_ph: observations.reshape(-1, self._observation_dim)}
        actions = tf.get_default_session().run(self._actions, feeds)
        return actions

    def get_params_internal(self,):
        scope = self._scope_name
            # Add "/" to 'scope' unless it's empty (otherwise get_collection will
            # return all parameters that start with 'scope'.

        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    def actions_for(self, observations, n_action_samples=1, reuse=False):
        n_state_samples = tf.shape(observations)[0]
        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (n_state_samples, n_action_samples,
                            3)
            
        else:
            latent_shape = (n_state_samples, 3)

        latents = tf.random_normal(latent_shape)
        joint_actions = []
        for i in range(3):
            with tf.variable_scope(self._name+'marg'+str(i), reuse=reuse):
                my_obs_dim = self._observation_dim//3 *(i+1)
                my_action_dim = self._action_dim//3 *(i+1)
                if n_action_samples > 1:
                    my_obs = tf.expand_dims(observations[:, :, i], axis=-1)
                    my_actions = tf.expand_dims(latents[:, :, i], axis=-1)
                else:
                    my_obs = tf.expand_dims(observations[:, i], axis=-1)
                    my_actions = tf.expand_dims(latents[:, i], axis=-1)
                joint_actions.append(feedforward_net(
                    (my_obs, my_actions),
                    layer_sizes=self._layer_sizes,
                    activation_fn=tf.nn.relu,
                    output_nonlinearity=None, name = self._name+'marg'+str(i))
                )
        my_action_pl = tf.concat(joint_actions, axis=-1)
        
        for j in range(self.nego_round):
            final_actions = []
            for i in range(3):
                with tf.variable_scope(self._name+'cond'+str(i), reuse=reuse):
                    my_obs_dim = self._observation_dim//3 *(i+1)
                    my_action_dim = self._action_dim//3 *(i+1)
                    if n_action_samples > 1:
                        my_obs = tf.expand_dims(observations[:, :, i], axis=-1)
                        my_actions = tf.expand_dims(latents[:, :, i], axis=-1)
                    else:
                        my_obs = tf.expand_dims(observations[:, i], axis=-1)
                        my_actions = tf.expand_dims(latents[:, i], axis=-1)
                    
                    final_actions.append(feedforward_net(
                        (my_obs, my_action_pl),
                        layer_sizes=self._layer_sizes,
                        activation_fn=tf.nn.relu,
                        output_nonlinearity=None, name = self._name+'cond'+str(i))
                    )
            my_action_pl = tf.concat(final_actions, axis=-1)
                    
        raw_actions = my_action_pl

        if self.sampling:
            u = tf.random_uniform(tf.shape(raw_actions))
            return tf.nn.softmax(raw_actions - tf.log(-tf.log(u)), axis=-1)

        if (self.shift is not None) and (self.scale is not None) and self._squash:
            tf.scalar_mul(self.scale, tf.tanh(raw_actions) + self.shift)
        print('stochastic', self._u_range, self._squash, self._squash_func)
        return tf.scalar_mul(self._u_range, self._squash_func(raw_actions)) if self._squash else tf.clip_by_value(raw_actions, -self._u_range, self._u_range)

class ACConditionedStochasticNNPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy."""

    def __init__(self,
                 env_spec=None,
                 observation_space=None,
                 action_space=None,
                 nego_round = 1,
                 hidden_layer_sizes=(100, 100),
                 squash=False,
                 squash_func=tf.tanh,
                 name='accstochastic_policy',
                 u_range=1.,
                 shift=None,
                 scale=None,
                 joint=False, agent_id=None, sampling=False):
        Serializable.quick_init(self, locals())
        if env_spec is None:
            self._observation_dim = observation_space
            self._action_dim = action_space
        elif isinstance(env_spec, MAEnvSpec):
            assert agent_id is not None
            self._observation_dim = env_spec.observation_space.flat_dim
            self._action_dim = env_spec.action_space.flat_dim
        else:
            self._action_dim = env_spec.action_space.flat_dim
            self._observation_dim = env_spec.observation_space.flat_dim **2
        self._layer_sizes = list(hidden_layer_sizes) + [1]
        self._squash = squash
        self._squash_func = squash_func
        self._u_range = u_range
        self.shift = shift
        self.scale = scale
        self._name = name + '_agent_{}'.format(agent_id)
        self.sampling = sampling
        self.agent_id = agent_id
        self.nego_round = nego_round

        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation_{}_agent_{}'.format(name, agent_id))
        # self._actions_ph = tf.placeholder(
        #     tf.float32,
        #     shape=[None, self._observation_dim],
        #     name='actions_{}_agent_{}'.format(name, agent_id))
        # self._observation_ph = None
        # self._actions = None

        self._actions = self.actions_for(self._observation_ph)
        
        super(ACConditionedStochasticNNPolicy, self).__init__(
            env_spec, self._observation_ph, self._actions, self._name)


    def get_actions(self, observations):
        feeds = {self._observation_ph: observations.reshape(-1, self._observation_dim)}
        actions = tf.get_default_session().run(self._actions, feeds)
        return actions

    def get_params_internal(self,):
        scope = self._scope_name
            # Add "/" to 'scope' unless it's empty (otherwise get_collection will
            # return all parameters that start with 'scope'.

        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    def actions_for(self, observations, n_action_samples=1, reuse=False):
        n_state_samples = tf.shape(observations)[0]
        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (n_state_samples, n_action_samples,
                            3)
            
        else:
            latent_shape = (n_state_samples, 3)

        latents = tf.random_normal(latent_shape)
        
        my_action_pl = latents
        
        for j in range(self.nego_round):
            final_actions = []
            for i in range(3):
                with tf.variable_scope(self._name+'cond'+str(i), reuse=reuse):
                    my_obs_dim = self._observation_dim//3 *(i+1)
                    my_action_dim = self._action_dim//3 *(i+1)
                    if n_action_samples > 1:
                        my_obs = tf.expand_dims(observations[:, :, i], axis=-1)
                        my_actions = tf.expand_dims(latents[:, :, i], axis=-1)
                    else:
                        my_obs = tf.expand_dims(observations[:, i], axis=-1)
                        my_actions = tf.expand_dims(latents[:, i], axis=-1)
                    
                    final_actions.append(feedforward_net(
                        (my_obs, my_action_pl),
                        layer_sizes=self._layer_sizes,
                        activation_fn=tf.nn.relu,
                        output_nonlinearity=None, name = self._name+'cond'+str(i))
                    )
            my_action_pl = tf.concat(final_actions, axis=-1)
                    
        raw_actions = my_action_pl

        if self.sampling:
            u = tf.random_uniform(tf.shape(raw_actions))
            return tf.nn.softmax(raw_actions - tf.log(-tf.log(u)), axis=-1)

        if (self.shift is not None) and (self.scale is not None) and self._squash:
            tf.scalar_mul(self.scale, tf.tanh(raw_actions) + self.shift)
        print('stochastic', self._u_range, self._squash, self._squash_func)
        return tf.scalar_mul(self._u_range, self._squash_func(raw_actions)) if self._squash else tf.clip_by_value(raw_actions, -self._u_range, self._u_range)

class ASSConditionedStochasticNNPolicy(NNPolicy, Serializable):
    """Stochastic neural network policy."""

    def __init__(self,
                 env_spec=None,
                 observation_space=None,
                 action_space=None,
                 hidden_layer_sizes=(100, 100),
                 squash=False,
                 squash_func=tf.tanh,
                 name='sscstochastic_policy',
                 u_range=1.,
                 shift=None,
                 scale=None,
                 joint=False, agent_id=None, sampling=False):
        Serializable.quick_init(self, locals())
        if env_spec is None:
            self._observation_dim = observation_space
            self._action_dim = action_space
        elif isinstance(env_spec, MAEnvSpec):
            assert agent_id is not None
            self._observation_dim = env_spec.observation_space.flat_dim
            self._action_dim = env_spec.action_space.flat_dim
        else:
            self._action_dim = env_spec.action_space.flat_dim
            self._observation_dim = env_spec.observation_space.flat_dim **2
        self._layer_sizes = list(hidden_layer_sizes) + [1]
        self._squash = squash
        self._squash_func = squash_func
        self._u_range = u_range
        self.shift = shift
        self.scale = scale
        self._name = name + '_agent_{}'.format(agent_id)
        self.sampling = sampling
        self.agent_id = agent_id

        self._observation_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._observation_dim],
            name='observation_{}_agent_{}'.format(name, agent_id))
        # self._actions_ph = tf.placeholder(
        #     tf.float32,
        #     shape=[None, self._observation_dim],
        #     name='actions_{}_agent_{}'.format(name, agent_id))
        # self._observation_ph = None
        # self._actions = None

        self._actions = self.actions_for(self._observation_ph)
        
        super(ASSConditionedStochasticNNPolicy, self).__init__(
            env_spec, self._observation_ph, self._actions, self._name)


    def get_actions(self, observations):
        feeds = {self._observation_ph: observations.reshape(-1, self._observation_dim)}
        actions = tf.get_default_session().run(self._actions, feeds)
        return actions

    def get_params_internal(self,):
        scope = self._scope_name
            # Add "/" to 'scope' unless it's empty (otherwise get_collection will
            # return all parameters that start with 'scope'.

        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    def actions_for(self, observations, n_action_samples=1, reuse=False):
        n_state_samples = tf.shape(observations)[0]
        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (n_state_samples, n_action_samples,
                            3)
            
        else:
            latent_shape = (n_state_samples, 3)

        latents = tf.random_normal(latent_shape)
        my_actions = []
        for i in range(3):
            with tf.variable_scope(self._name+str(i), reuse=reuse):
                my_obs_dim = self._observation_dim//3 *(i+1)
                my_action_dim = self._action_dim//3 *(i+1)
                if n_action_samples > 1:
                    my_obs = observations[:, :, :my_obs_dim]
                    my_lantents = latents[:, :, :my_obs_dim]
                else:
                    my_obs = observations[:, :my_obs_dim]
                    my_lantents = latents[:, :my_obs_dim]
                
                my_actions.append(feedforward_net(
                    (my_obs, my_lantents),
                    layer_sizes=self._layer_sizes,
                    activation_fn=tf.nn.relu,
                    output_nonlinearity=None, name=self._name+str(i))
                )
        raw_actions = tf.concat(my_actions, -1)

        if self.sampling:
            u = tf.random_uniform(tf.shape(raw_actions))
            return tf.nn.softmax(raw_actions - tf.log(-tf.log(u)), axis=-1)

        if (self.shift is not None) and (self.scale is not None) and self._squash:
            tf.scalar_mul(self.scale, tf.tanh(raw_actions) + self.shift)
        print('stochastic', self._u_range, self._squash, self._squash_func)
        return tf.scalar_mul(self._u_range, self._squash_func(raw_actions)) if self._squash else tf.clip_by_value(raw_actions, -self._u_range, self._u_range)
        