import numpy as np
import argparse

from maci.learners import MAVBAC, MASQL, ROMMEO, JSQL, GPF, GPF_SS
from maci.misc.sampler import MASampler, JSampler, SSampler
from maci.environments import PBeautyGame, MatrixGame, DifferentialGame, JDifferentialGame
from maci.environments import make_particle_env
from maci.misc import logger
import gtimer as gt
import datetime
from copy import deepcopy
from maci.get_agents import ddpg_agent, masql_agent, pr2ac_agent, rom_agent, jsql_agent, gpf_agent, jsql_ss_agent, gpf_ss_agent, ccf_ss_agent, accf_ss_agent, agpf_ss_agent

import pickle 

import maci.misc.tf_utils as U
import os
import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

from keras.backend.tensorflow_backend import set_session, clear_session
import tensorflow as tf
from easy_tf_log import tflog
import easy_tf_log



def get_particle_game(particle_game_name, arglist):
    env = make_particle_env(game_name=particle_game_name)
    print(env.action_space, env.observation_space)
    agent_num = env.n
    adv_agent_num = 0
    if particle_game_name == 'simple_push' or particle_game_name == 'simple_adversary':
        adv_agent_num = 1
    elif particle_game_name == 'simple_tag':
        adv_agent_num = 3
    model_names_setting = arglist.model_names_setting.split('_')
    model_name = '_'.join(model_names_setting)
    model_names = [model_names_setting[1]] * adv_agent_num + [model_names_setting[0]] * (agent_num - adv_agent_num)
    return env, agent_num, model_name, model_names



def objective(arglist):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)
    game_name = arglist.game_name
    # 'abs', 'one'
    reward_type = arglist.reward_type
    p = arglist.p
    agent_num = arglist.n
    u_range = 1.
    k = 0
    print(arglist.aux, 'arglist.aux')
    model_names_setting = arglist.model_names_setting.split('_')
    model_names = [model_names_setting[0]]
    model_name = '_'.join(model_names)
    path_prefix = game_name
    if game_name == 'pbeauty':
        env = PBeautyGame(agent_num=agent_num, reward_type=reward_type, p=p)
        path_prefix  = game_name + '-' + reward_type + '-' + str(p)
    elif 'matrix' in game_name:
        matrix_game_name = game_name.split('-')[-1]
        repeated = arglist.repeat
        max_step = arglist.max_path_length
        memory = arglist.memory
        env = MatrixGame(game=matrix_game_name, agent_num=agent_num,
                         action_num=2, repeated=repeated,
                         max_step=max_step, memory=memory,
                         discrete_action=False, tuple_obs=False)
        path_prefix = '{}-{}-{}-{}'.format(game_name, repeated, max_step, memory)

    elif 'diff' in game_name:
        diff_game_name = game_name.split('-')[-1]
        
        s2 = arglist.s2
        x2 = arglist.x2
        y2 = arglist.y2
        con = arglist.con
        env = DifferentialGame(diff_game_name, 3, x2, y2, s2, con)
        agent_num = 1

    elif 'particle' in game_name:
        particle_game_name = game_name.split('-')[-1]
        env, agent_num, model_name, model_names = get_particle_game(particle_game_name, arglist)

    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
    if 'CG' in model_name:
        model_name = model_name + '-{}'.format(arglist.mu)
    if not arglist.aux:
        model_name = model_name + '-{}'.format(arglist.aux)

    suffix = '{}/{}/{}/{}'.format(path_prefix, agent_num, model_name, timestamp)

    print(suffix)

    # logger.add_tabular_output('./log/{}.csv'.format(suffix))
    # snapshot_dir = './snapshot/{}'.format(suffix)
    # policy_dir = './policy/{}'.format(suffix)
    # os.makedirs(snapshot_dir, exist_ok=True)
    # os.makedirs(policy_dir, exist_ok=True)
    # logger.set_snapshot_dir(snapshot_dir)

    agents = []
    M = arglist.hidden_size
    batch_size = arglist.batch_size

    sampler = SSampler(agent_num=agent_num, joint=True, global_reward=arglist.global_reward, max_path_length=25, min_pool_size=100, batch_size=batch_size)

    base_kwargs = {
        'sampler': sampler,
        'epoch_length': 1,
        'n_epochs': arglist.max_steps,
        'n_train_repeat': 1,
        'eval_render': True,
        'eval_n_episodes': 10
    }
    _alpha = arglist.alpha
    lr = arglist.lr
    n_pars = arglist.n_pars
    nego_round = arglist.nego_round
    result = 0.

    with U.single_threaded_session():
        for i, model_name in enumerate(model_names):
            if 'PR2AC' in model_name:
                k = int(model_name[-1])
                g = False
                mu = arglist.mu
                if 'G' in model_name:
                    g = True
                agent = pr2ac_agent(model_name, i, env, M, u_range, base_kwargs,  lr=lr, n_pars=n_pars, k=k, g=g, mu=mu, game_name=game_name, aux=arglist.aux)
            elif model_name == 'MASQL':
                agent = masql_agent(model_name, i, env, M, u_range, base_kwargs,  lr=lr, n_pars=n_pars, game_name=game_name)
            elif model_name == 'JSQL':
                agent = jsql_ss_agent(model_name, i, env, M, u_range, base_kwargs,  lr=lr, n_pars=n_pars, game_name=game_name)
            elif model_name == 'GPF':
                agent = gpf_ss_agent(model_name, i, env, M, u_range, base_kwargs,  lr=lr, n_pars=n_pars, batch_size = batch_size, game_name=game_name)
            elif model_name == 'AGPF':
                agent = agpf_ss_agent(model_name, i, env, M, u_range, base_kwargs,  lr=lr, n_pars=n_pars, batch_size = batch_size, game_name=game_name)
            elif model_name == 'CCF':
                agent = ccf_ss_agent(model_name, i, env, M, u_range, base_kwargs, nego_round=nego_round, lr=lr, n_pars=n_pars, batch_size = batch_size, game_name=game_name)
            elif model_name == 'ACCF':
                agent = accf_ss_agent(model_name, i, env, M, u_range, base_kwargs, nego_round=nego_round, lr=lr, n_pars=n_pars, batch_size = batch_size, game_name=game_name)
            elif model_name == 'ROMMEO':
                agent = rom_agent(model_name, i, env, M, u_range, base_kwargs, game_name=game_name)
            else:
                if model_name == 'DDPG':
                    joint = False
                    opponent_modelling = False
                elif model_name == 'MADDPG':
                    joint = True
                    opponent_modelling = False
                elif model_name == 'DDPG-OM':
                    joint = True
                    opponent_modelling = True
                agent = ddpg_agent(joint, opponent_modelling, model_names, i, env, M, u_range, base_kwargs,lr=lr, game_name=game_name)

            agents.append(agent)

        sampler.initialize(env, agents)

        for agent in agents:
            agent._init_training()
        gt.rename_root('MARLAlgorithm')
        gt.reset()
        gt.set_def_unique(False)
        initial_exploration_done = False
        # noise = .1
        noise = .5


        for agent in agents:
            try:
                agent.policy.set_noise_level(noise)
            except:
                pass
        # alpha = .5
        for steps in gt.timed_for(range(base_kwargs['n_epochs'] + 1)):
            # import pdb; pdb.set_trace()
            # 
            # if steps > 100 and steps<150: 
            #     alpha = .1 - 0.099 * steps/(150)
            # elif steps >= 150:
            #     alpha = 1e-3
            if steps < base_kwargs['n_epochs']//3:
                # alpha = _alpha
                alpha = _alpha + np.exp(-0.1 * max(steps-10, 0)) * 500.
            elif  steps < base_kwargs['n_epochs']//2:
                alpha = _alpha/10
            else:
                alpha = .3
            tflog('alpha', alpha)
            print('alpha', alpha)
            # logger.push_prefix('Epoch #%d | ' % steps)
            if steps % (25*1000) == 0:
                print(suffix)
            for t in range(base_kwargs['epoch_length']):
                # TODO.code consolidation: Add control interval to sampler
                if not initial_exploration_done:
                    # if steps >= 1000:
                    if steps >= 10:
                        initial_exploration_done = True
                sampler.sample()
                if not initial_exploration_done:
                    continue
                gt.stamp('sample')
                print('Sample Done')
                if steps == 10000:
                    noise = 0.1

                    for agent in agents:
                        try:
                            agent.policy.set_noise_level(noise)
                        except:
                            pass
                    # alpha = 10.
                # if steps == 2000:
                if steps > base_kwargs['n_epochs'] / 10:
                    noise = 0.1
                    for agent in agents:
                        try:
                            agent.policy.set_noise_level(noise)
                        except:
                            pass
                    # alpha = .1
                if steps > base_kwargs['n_epochs'] / 5:
                    noise = 0.05
                    for agent in agents:
                        try:
                            agent.policy.set_noise_level(noise)
                        except:
                            pass
                if steps > base_kwargs['n_epochs'] / 6:
                    noise = 0.01
                    for agent in agents:
                        try:
                            agent.policy.set_noise_level(noise)
                        except:
                            pass
                if steps % arglist.training_interval != 0:
                    continue
                for j in range(base_kwargs['n_train_repeat']):
                    batch_n = []
                    recent_batch_n = []
                    indices = None
                    receent_indices = None
                    for i, agent in enumerate(agents):
                        if i == 0:
                            batch = agent.pool.random_batch(batch_size)
                            indices = agent.pool.indices
                            receent_indices = list(range(agent.pool._top-batch_size, agent.pool._top))

                        batch_n.append(agent.pool.random_batch_by_indices(indices))
                        recent_batch_n.append(agent.pool.random_batch_by_indices(receent_indices))

                    # print(len(batch_n))
                    target_next_actions_n = []
                    # try:
                    all_obs = np.array(np.concatenate([batch['observations'] for batch in batch_n], axis=-1))
                    all_next_obs = np.array(np.concatenate([batch['next_observations'] for batch in batch_n], axis=-1))
                    # print(all_obs[0])
                    for batch in batch_n:
                        # print('making all obs')
                        batch['all_observations'] = deepcopy(all_obs)
                        batch['all_next_observations'] = deepcopy(all_next_obs)
                    # opponent_current_actions_n = []
                    # for agent, batch in zip(agents, batch_n):
                    #     target_next_actions_n.append(agent.target_policy.get_actions(batch['next_observations']))
                        # opponent_current_actions_n.append(agent.policy.get_actions(batch['observations']))

                    # for i, agent in enumerate(agents):
                    #     batch_n[i]['opponent_current_actions'] = np.reshape(
                    #         np.delete(deepcopy(opponent_current_actions_n), i, 0), (-1, agent._opponent_action_dim))
                    # except:
                    #     pass


                    # opponent_actions_n = np.array([batch['actions'] for batch in batch_n])
                    # recent_opponent_actions_n = np.array([batch['actions'] for batch in recent_batch_n])

                    ####### figure out
                    # recent_opponent_observations_n = []
                    # for batch in recent_batch_n:
                    #     recent_opponent_observations_n.append(batch['observations'])


                    # current_actions = [agents[i].policy.get_actions(batch_n[i]['next_observations'])[0][0] for i in range(agent_num)]
                    # all_actions_k = []
                    # for i, agent in enumerate(agents):
                    #     if isinstance(agent, MAVBAC):
                    #         if agent._k > 0:
                    #             batch_actions_k = agent.policy.get_all_actions(batch_n[i]['next_observations'])
                    #             actions_k = [a[0][0] for a in batch_actions_k]
                    #             all_actions_k.append(';'.join(list(map(str, actions_k))))
                    # if len(all_actions_k) > 0:
                    #     with open('{}/all_actions.csv'.format(policy_dir), 'a') as f:
                    #         f.write(','.join(list(map(str, all_actions_k))) + '\n')
                    # with open('{}/policy.csv'.format(policy_dir), 'a') as f:
                    #     f.write(','.join(list(map(str, current_actions)))+'\n')
                    # print('============')
                    for i, agent in enumerate(agents):
                        # try:
                        #     batch_n[i]['next_actions'] = deepcopy(target_next_actions_n[i])
                        # except:
                        #     pass
                        # batch_n[i]['opponent_actions'] = np.reshape(np.delete(deepcopy(opponent_actions_n), i, 0), (-1, agent._opponent_action_dim))
                        # if agent.joint:
                        #     if agent.opponent_modelling:
                        #         batch_n[i]['recent_opponent_observations'] = recent_opponent_observations_n[i]
                        #         batch_n[i]['recent_opponent_actions'] = np.reshape(np.delete(deepcopy(recent_opponent_actions_n), i, 0), (-1, agent._opponent_action_dim))
                        #         batch_n[i]['opponent_next_actions'] = agent.opponent_policy.get_actions(batch_n[i]['next_observations'])
                        #     else:
                        #         batch_n[i]['opponent_next_actions'] = np.reshape(np.delete(deepcopy(target_next_actions_n), i, 0), (-1, agent._opponent_action_dim))

                        # if isinstance(agent, MAVBAC) or isinstance(agent, MASQL) or isinstance(agent, ROMMEO):
                        agent._do_training(iteration=t + steps * agent._epoch_length, batch=batch_n[i], annealing=alpha)
                        # else:
                        #     agent._do_training(iteration=t + steps * agent._epoch_length, batch=batch_n[i])
                gt.stamp('train')
            result = sampler.terminate()
            print('res is', result)
    clear_session()
    with open('step_act.pickle', 'wb') as handle:
        pickle.dump(sampler.step_act_dict, handle)
    with open('step_rew.pickle', 'wb') as handle:
        pickle.dump(sampler.step_rew_dict, handle)
        
    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', "--game_name", type=str, default="diff-ma_softq", help="name of the game")


    # Differential game settings
    parser.add_argument('-alpha', "--alpha", type=float, default=1)
    parser.add_argument('-lr', "--lr", type=float, default=1)
    parser.add_argument('-n_pars', "--n_pars", type=int, default=1)
    parser.add_argument('-s2', "--s2", type=float, default=1)
    parser.add_argument('-x2', "--x2", type=float, default=7)
    parser.add_argument('-y2', "--y2", type=float, default=7)
    parser.add_argument('-c', "--con", type=float, default=10.0)

    parser.add_argument('-nego_round', "--nego_round", type=int, default=1)
    parser.add_argument('-p', "--p", type=float, default=1.1, help="p")
    parser.add_argument('-mu', "--mu", type=float, default=1.5, help="mu")
    parser.add_argument('-r', "--reward_type", type=str, default="abs", help="reward type")
    parser.add_argument('-mp', "--max_path_length", type=int, default=1, help="reward type")
    parser.add_argument('-ms', "--max_steps", type=int, default=500, help="reward type")
    parser.add_argument('-me', "--memory", type=int, default=0, help="reward type")
    parser.add_argument('-n', "--n", type=int, default=2, help="name of the game")
    parser.add_argument('-do_nego', "--do_nego", type=int, default=1, help="name of the game")
    parser.add_argument('-bs', "--batch_size", type=int, default=512, help="name of the game")
    parser.add_argument('-hm', "--hidden_size", type=int, default=100, help="name of the game")
    parser.add_argument('-ti', "--training_interval", type=int, default=1, help="name of the game")
    parser.add_argument('-re', "--repeat", type=bool, default=False, help="name of the game")
    parser.add_argument('-a', "--aux", type=bool, default=True, help="name of the game")
    parser.add_argument('-gr', "--global_reward", type=bool, default=False, help="name of the game")
    parser.add_argument('-m', "--model_names_setting", type=str, default='MADDPG_MADDPG', help="models setting agent vs adv")

    arglist = parser.parse_args()
    objective(arglist)
    # study_name = 'N3D-ss-' + str(arglist.max_steps) + '-' + arglist.model_names_setting + str(arglist.do_nego)+ '(' + str(arglist.x2) + ',' + str(arglist.y2) + ',' + str(arglist.s2) + ')'
    # storage_name = "mysql://jjs@localhost/{}".format('diff_param')
    # study = optuna.create_study(study_name=study_name, storage=storage_name, directions=["maximize"], load_if_exists=True)
    # # study = optuna.create_study(study_name=study_name, directions=["maximize"])
    # study.optimize(lambda trial: objective(trial, arglist), n_trials=50)

    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # print("Study statistics: ")
    # print("Number of finished trials: ", len(study.trials))
    # print("Number of complete trials: ", len(complete_trials))

    # print("Best trial:")
    # trial = study.best_trials