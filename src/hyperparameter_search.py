import const
from src import agent, algo, utils_env

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import ParameterGrid

state_size = const.state_size
action_size = const.action_size
N = const.rolling_mean_N


class MyNavigator(BaseEstimator, ClassifierMixin):
    def __init__(self, num_states: int = const.state_size,
                 num_actions: int = const.action_size,
                 num_episodes: int = const.num_episodes,
                 memory_size: int = const.memory_size,
                 gamma: float = const.gamma,
                 batch_size: int = const.batch_size,
                 expl_noise: int = const.expl_noise,
                 model_learning_rate: float = const.model_learning_rate,
                 num_fc_actor: int = const.num_fc_actor,
                 num_fc_critic: int = const.num_fc_critic
                 ):

        # algo params
        self.num_episodes = num_episodes

        # agent params
        self.num_states = num_states
        self.num_actions = num_actions
        self.memory_size = memory_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.expl_noise = expl_noise

        # model params
        self.model_learning_rate = model_learning_rate
        self.num_fc_actor = num_fc_actor
        self.num_fc_critic = num_fc_critic

    def fit(self, i: int, env: utils_env.Environment):
        self.ag = agent.DRLAgent(state_size, action_size,
                                 self.memory_size, self.gamma, self.batch_size,
                                 self.expl_noise, self.model_learning_rate,
                                 self.num_fc_actor, self.num_fc_critic)
        self.ag.set_model_path(i)  # save each candidate's model separately

        self.al = algo.DRLAlgo(env, self.ag, self.num_episodes)
        self.al.set_image_path(i)  # save each candidate's score separately

        history = self.al.train(with_close=False)  # do not close the Env so that other agents can be trained
        score = self._get_score(history)
        return score

    def _get_score(self, hist):
        # prepare data
        x = pd.Series(hist)
        y = x.rolling(window=N).mean().iloc[N - 1:]
        if not y.empty:
            score = y.iloc[-1]
        else:
            score = 0.
        print('\n', score, hist)
        return score


def grid_search():
    env = utils_env.Environment()

    print('=' * 30, 'Grid Search', '=' * 30)

    params = {
        'num_episodes': [200, 250, 500],  # --> 250
        'batch_size': [32, 64, 128, 256],  # --> 32
        'expl_noise': [0.1, 0.3],  # --> 0.3
        'gamma': [0.95, 0.99],  # --> 0.95
        'model_learning_rate': [0.001, 0.0001, 0.00001],  # --> 0.001
        'num_fc_actor': [128, 64, 32],  # --> 128
        'num_fc_critic': [128, 64, 32],  # --> 128
        'memory_size': [20000, 40000]  # --> 40000
    }

    grid = ParameterGrid(params)
    rf = MyNavigator()

    best_score = -10.
    best_grid = None
    best_grid_index = 0
    result_dict = {}
    key_list = list(params.keys()) + ['score']
    df = pd.DataFrame(columns=key_list)

    for i, g in enumerate(grid):
        rf.set_params(**g)
        score = rf.fit(i, env)
        result_dict[i] = {'score': score, 'grid': g}

        d = g
        d['score'] = score
        df = df.append(d, ignore_index=True)

        print('Evaluated candidate:', i, result_dict[i])
        # save if best
        if score >= best_score:
            best_score = score
            best_grid = g
            best_grid_index = i

    for k, v in result_dict.items():
        print(k, v)

    print("==> Best score:", best_score)
    print("==> Best grid:", best_grid_index, best_grid)

    if len(key_list) == 3:  # better overview as pivot table (only for 2 hyperparams)
        for c in params.keys():  # if one hyperparam is a list of values
            if df[c].dtype == object:
                df[c] = df[c].astype(str)

        print(df.pivot(index=key_list[0], columns=key_list[1], values=key_list[2]))
    else:
        print(df)

    env.close()  # finally, close the Env
