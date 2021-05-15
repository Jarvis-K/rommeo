import numpy as np
from maci.misc.space import MADiscrete, MABox
from maci.environments.env_spec import MAEnvSpec


from rllab.core.serializable import Serializable


class DifferentialGame(Serializable):
    def __init__(self, game_name, agent_num, x2, y2, s2, con, action_low=-10, action_high=10):
        Serializable.quick_init(self, locals())
        self.game = game_name
        self.agent_num = agent_num
        # self.action_num = action_num
        self.action_range = [action_low, action_high]
        lows = np.array([np.array([action_low]) for _ in range(self.agent_num)])
        highs = np.array([np.array([action_high]) for _ in range(self.agent_num)])
        self.action_spaces = MABox(lows=lows, highs=highs)
        self.observation_spaces = MADiscrete([1] * self.agent_num)
        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)
        self.t = 0
        self.numplots = 0
        self.payoff = {}

        if self.game == 'zero_sum':
            assert self.agent_num == 2
            self.payoff[0] = lambda a1, a2: a1 * a2
            self.payoff[1] = lambda a1, a2: -a1 * a2
        elif self.game == 'trigonometric':
            assert self.agent_num == 2
            self.payoff[0] = lambda a1, a2: np.cos(a2) * a1
            self.payoff[1] = lambda a1, a2: np.sin(a1) * a2
        elif self.game == 'mataching_pennies':
            assert self.agent_num == 2
            self.payoff[0] = lambda a1, a2: (a1-0.5)*(a2-0.5)
            self.payoff[1] = lambda a1, a2: (a1-0.5)*(a2-0.5)
        elif self.game == 'rotational':
            assert self.agent_num == 2
            self.payoff[0] = lambda a1, a2: 0.5 * a1 * a1 + 10 * a1 * a2
            self.payoff[1] = lambda a1, a2: 0.5 * a2 * a2 - 10 * a1 * a2
        elif self.game == 'wolf':
            assert self.agent_num == 2
            def V(alpha, beta, payoff):
                u = payoff[(0, 0)] - payoff[(0, 1)] - payoff[(1, 0)] + payoff[(1, 1)]
                return alpha * beta * u + alpha * (payoff[(0, 1)] - payoff[(1, 1)]) + beta * (
                            payoff[(1, 0)] - payoff[(1, 1)]) + payoff[(1, 1)]

            payoff_0 = np.array([[0, 3], [1, 2]])
            payoff_1 = np.array([[3, 2], [0, 1]])

            self.payoff[0] = lambda a1, a2: V(a1, a2, payoff_0)
            self.payoff[1] = lambda a1, a2: V(a1, a2, payoff_1)

        elif self.game == 'ma_softq':
            h1 = 0.8
            h2 = 1.
            s1 = 3.
            s2 = s2
            x1 = -5.
            x2 = x2
            y1 = -5.
            y2 = y2
            z1 = 3
            z2 = -3

            c = con
            # c = 0.

            def max_f(a1, a2, a3):
                f1 = h1 * (-(np.square(a1 - x1) / s1) - (np.square(a2 - y1) / s1) - (np.square(a3 - z1) / s1))
                f2 = h2 * (-(np.square(a1 - x2) / s2) - (np.square(a2 - y2) / s2) - (np.square(a3 - z2) / s2)) + c

                return max(f1, f2)
            self.payoff[0] = lambda a1, a2, a3: max_f(a1, a2, a3)
            self.payoff[1] = lambda a1, a2, a3: max_f(a1, a2, a3)
            self.payoff[2] = lambda a1, a2, a3: max_f(a1, a2, a3)
        self.rewards = np.zeros((self.agent_num,))

    @staticmethod
    def get_game_list():
        return {
            'zero_sum': {'agent_num': 2, 'action_num': 2}
        }

    def step(self, actions):
        print('actions', actions)
        actions = np.array(actions).reshape((self.agent_num,)) * self.action_range[1]
        print('scaled', actions)
        reward_n = np.zeros((self.agent_num,))
        for i in range(self.agent_num):
            print('actions', actions)
            reward_n[i] = self.payoff[i](*tuple(actions))
        self.rewards = reward_n
        print(reward_n)
        state_n = np.array(list([[0. * i] for i in range(self.agent_num)]))
        # state_n = np.eyes(self.agent_num)
        info = {}
        done_n = np.array([True] * self.agent_num)
        self.t += 1
        return state_n, reward_n, done_n, info

    def reset(self):
        return np.array(list([[0. * i] for i in range(self.agent_num)]))
        # import pdb; pdb.set_trace()
        # return np.eyes(self.agent_num)

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(self.__str__())

    def get_joint_reward(self):
        return self.rewards

    def terminate(self):
        pass

    def __str__(self):
        content = 'Game Name {}, Number of Agent {}, Action Range {}\n'.format(self.game, self.agent_num, self.action_range)
        return content

class JDifferentialGame(Serializable):
    def __init__(self, game_name, agent_num, x2, y2, s2, con, action_low=-10, action_high=10):
        Serializable.quick_init(self, locals())
        self.game = game_name
        self.agent_num = 1
        self.n_dim = agent_num
        # self.action_num = action_num
        self.action_range = [action_low, action_high]
        lows = np.array([np.array([action_low]) for _ in range(self.agent_num)])
        highs = np.array([np.array([action_high]) for _ in range(self.agent_num)])
        self.action_spaces = MABox(lows=lows[0], highs=highs[0], shapes=[(self.n_dim)])
        self.observation_spaces = MADiscrete([self.n_dim] * self.agent_num)
        self.env_specs = MAEnvSpec(self.observation_spaces, self.action_spaces)
        self.t = 0
        self.numplots = 0
        self.payoff = {}

        if self.game == 'zero_sum':
            assert self.agent_num == 2
            self.payoff[0] = lambda a1, a2: a1 * a2
            self.payoff[1] = lambda a1, a2: -a1 * a2
        elif self.game == 'trigonometric':
            assert self.agent_num == 2
            self.payoff[0] = lambda a1, a2: np.cos(a2) * a1
            self.payoff[1] = lambda a1, a2: np.sin(a1) * a2
        elif self.game == 'mataching_pennies':
            assert self.agent_num == 2
            self.payoff[0] = lambda a1, a2: (a1-0.5)*(a2-0.5)
            self.payoff[1] = lambda a1, a2: (a1-0.5)*(a2-0.5)
        elif self.game == 'rotational':
            assert self.agent_num == 2
            self.payoff[0] = lambda a1, a2: 0.5 * a1 * a1 + 10 * a1 * a2
            self.payoff[1] = lambda a1, a2: 0.5 * a2 * a2 - 10 * a1 * a2
        elif self.game == 'wolf':
            assert self.agent_num == 2
            def V(alpha, beta, payoff):
                u = payoff[(0, 0)] - payoff[(0, 1)] - payoff[(1, 0)] + payoff[(1, 1)]
                return alpha * beta * u + alpha * (payoff[(0, 1)] - payoff[(1, 1)]) + beta * (
                            payoff[(1, 0)] - payoff[(1, 1)]) + payoff[(1, 1)]

            payoff_0 = np.array([[0, 3], [1, 2]])
            payoff_1 = np.array([[3, 2], [0, 1]])

            self.payoff[0] = lambda a1, a2: V(a1, a2, payoff_0)
            self.payoff[1] = lambda a1, a2: V(a1, a2, payoff_1)

        elif self.game == 'ma_softq':
            h1 = 0.8
            h2 = 1.
            s1 = 3.
            s2 = s2
            x1 = -5.
            x2 = x2
            y1 = -5.
            y2 = y2
            c = con
            z1 = 3
            z2 = -3

            c = con
            # c = 0.

            def max_f(a1, a2, a3):
                f1 = h1 * (-(np.square(a1 - x1) / s1) - (np.square(a2 - y1) / s1) - (np.square(a3 - z1) / s1))
                f2 = h2 * (-(np.square(a1 - x2) / s2) - (np.square(a2 - y2) / s2) - (np.square(a3 - z2) / s2)) + c

                return max(f1, f2)
            self.payoff[0] = lambda a1, a2, a3: max_f(a1, a2, a3)
            self.payoff[1] = lambda a1, a2, a3: max_f(a1, a2, a3)
        self.rewards = np.zeros((self.agent_num,))

    @staticmethod
    def get_game_list():
        return {
            'zero_sum': {'agent_num': 2, 'action_num': 2}
        }

    def step(self, actions):
        assert len(actions) == self.n_dim
        print('actions', actions)
        actions = np.array(actions).reshape((-1)) * self.action_range[1]
        print('scaled', actions)
        reward_n = np.zeros((self.agent_num,))
        for i in range(self.agent_num):
            reward_n[i] = self.payoff[i](*tuple(actions))
        self.rewards = reward_n
        print(reward_n)
        # state_n = np.array(list([[float(j) for j in range(self.n_dim)] for i in range(self.agent_num)]))
        state_n = np.eye(self.n_dim)
        info = {}
        done_n = np.array([True] * self.agent_num)
        self.t += 1
        return state_n, reward_n, done_n, info

    def reset(self):
        # return np.array(list([[float(j) for j in range(self.n_dim)] for i in range(self.agent_num)]))
        return np.eye(self.n_dim)

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(self.__str__())

    def get_joint_reward(self):
        return self.rewards

    def terminate(self):
        pass

    def __str__(self):
        content = 'Game Name {}, Number of Agent {}, Action Range {}\n'.format(self.game, self.agent_num, self.action_range)
        return content


if __name__ == '__main__':
    print(DifferentialGame.get_game_list())
    game = DifferentialGame('zero_sum', agent_num=2)
    print(game)