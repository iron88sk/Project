"""
Particular class of Seoul

"""
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import configparser
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
from envs.env import PhaseMap, PhaseSet, TrafficSimulator
from small_grid.data.build_file import gen_rou_file

sns.set_color_codes()

STATE_NAMES = ['wave', 'wait']

# map from ild order (alphabeta) to signal order (clockwise from north)

NODES =  {'ag1': ('2.0', ['ag7', 'ag2']),
          'ag2': ('6.0', ['ag1', 'ag8', 'ag3']),
          'ag3': ('4.0', ['ag2', 'ag9', 'ag4']),
          'ag4': ('2.1', ['ag3', 'ag5']),
          'ag5': ('2.2', ['ag4', 'ag10']),
          'ag7': ('2.4', ['ag11', 'ag8', 'ag1']),
          'ag8': ('2.5', ['ag7', 'ag13', 'ag9', 'ag2']),
          'ag9': ('2.6', ['ag8', 'ag14', 'ag10', 'ag3']),
          'ag10': ('2.7', ['ag9', 'ag15', 'ag5']),
          'ag11': ('2.8', ['ag13', 'ag7']),
          'ag13': ('3.1', ['ag11', 'ag14', 'ag8']),
          'ag14': ('4.1', ['ag13', 'ag15', 'ag9']),
          'ag15': ('3.2', ['ag14', 'ag10'])}

PHASES = {'2.0': ['GGGGGGGrrr', 'rrrrGGGGGG'],
          '2.1': ['GGGGrrGGGGrrr', 'rrrrGGrrrrGGG'],
          '2.2': ['GGGGrrrGGG', 'rrrrGGGrrr'],
          '2.3': ['GGGGGG', 'rrrrrr'],
          '2.4': ['rrGGGgrr', 'GGrrrrGg'],
          '2.5': ['GGGGGrrrGGGGrrr', 'rrrrrGGgrrrrGGg'],
          '2.6': ['GGGrrrrGGGGrrrr', 'rrrGGgrrrrrgGGg'],
          '2.7': ['GGGrrrrrGGg', 'rrrGGGGGrrr'],
          '2.8': ['gGGGGrrrrGG', 'grrrrGGGGrr'],
          '3.1': ['rrrGGgrrrrr', 'GGGrrrGGGrr', 'rrrGGrrrrGG'],
          '3.2': ['GGGGrrrGGrrrr', 'rrrrGGrrrrrrr', 'rrrrrrGrrGGGg'],
          '4.0': ['GGGGrrrrrrrGGGGGrrrrrr', 'rrrrrrrrrrrGGGGGGGrrrr', 'rrrrGGGrrrrrrrrrGGrrrr', 'rrrrrrrGGGGrrrrrrrGGGG'],
          '4.1': ['GGGrrrrrrrGGGrrrr', 'rrrGGrrrrrrrrGrrr', 'rrrrrGGGGrrrrrGGG', 'rrrrrrrrrGrrrrrrr'],
          '6.0': ['rrrrGGGGrrrrrrGGGGGr', 'rrrrrrrrrrrrrrGGGGGG', 'rrrrrrrrGrrrrrrrrrrG', 'GGrrrrrrrGGGrrrrrrrr', 'GGGGrrrrrrrrrrrrrrrr', 'rrGGrrrrrrrrGGrrrrrr']}

SECONDARY_ILDS_MAP = {'-218923156#0_0': '-218923156#1_1', '-218923156#0_1': '-218923156#1_2', '-218923156#0_2': '-218923156#1_3', '-218923156#0_3': '-218923156#1_4',
                     '-218923156#0_4': '-218923156#1_5', '-219053199#5_0': '-219053199#7_0', '-219053199#5_1': '-219053199#7_1', '-219053199#5_2': '-219053199#7_2',
                     '-420361195#3_0': '-329995230#0_0', '-420361195#3_1': '-329995230#0_0', '184083199#6_0': '184083199#0.274_0', '184083199#6_1':'184083199#0.274_1',
                     '184083199#6_2': '184083199#0.274_2', '218923160#4_0': '218923160#0-AddedOffRampEdge_1', '218923160#4_1': '218923160#0-AddedOffRampEdge_2',
                     '218923160#4_2': '218923160#0-AddedOffRampEdge_3', '219053199#3_0': '219053199#0_1', '219053199#3_1': '219053199#0_2', '219053199#3_2': '219053199#0_3',
                     '350127794#5_0': '350127794#4_0', 'gneE28_0': '469260694#6_0', 'gneE28_1': '469260694#6_1', '474532059#5_0': '474532059#4.95_0',
                     '474532059#5_1': '474532059#4.95_1', '782892488#3_0': '782892488#2_0', '782892488#3_1': '782892488#2_1', '782892488#3_2': '782892488#2_2',
                     '919117822#4_0': '919117822#3_0', '919117822#4_1': '919117822#3_1', '919117822#4_2': '919117822#3_1', '219096265_0': 'gneE3_0', '219096265_1': 'gneE3_1',
                     '219096265_2': 'gneE3_2', '219096265_3': 'gneE3_3', 'gneE36_0': 'gneE40_0', 'gneE36_1': 'gneE40_1', 'gneE36_2': 'gneE40_2', '656122981#2_1': 'gneE41_0',
                     '656122981#2_2': 'gneE41_1', '656122981#2_3': 'gneE41_2'}
class SeoulPhase(PhaseMap):
    def __init__(self):
        self.phases = {}
        for key, val in PHASES.items():
            self.phases[key] = PhaseSet(val)


class  SeoulController:
    def __init__(self, node_names, nodes):
        self.name = 'greedy'
        self.node_names = node_names
        self.nodes = nodes

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions

    def greedy(self, ob, node_name):
        # get the action space
        phases = PHASES[NODES[node_name][0]]
        flows = []
        node = self.nodes[node_name]
        # get the green waves
        for phase in phases:
            wave = 0
            visited_ilds = set() 
            for i, signal in enumerate(phase):
                if signal == 'G':
                    # find controlled lane
                    lane = node.lanes_in[i]
                    # ild = 'ild:' + lane
                    ild = lane
                    # if it has not been counted, add the wave
                    if ild not in visited_ilds:
                        j = node.ilds_in.index(ild)
                        wave += ob[j]
                        visited_ilds.add(ild)
            flows.append(wave)
        return np.argmax(np.array(flows))


class SeoulEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.num_car_hourly = config.getint('num_extra_car_per_hour')
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _get_node_phase_id(self, node_name):
        return self.phase_node_map[node_name]

    def _init_neighbor_map(self):
        return dict([(key, val[1]) for key, val in NODES.items()])

    def _init_map(self):
        self.neighbor_map = self._init_neighbor_map()
        self.phase_map = SeoulPhase()
        self.phase_node_map = dict([(key, val[0]) for key, val in NODES.items()])
        self.state_names = STATE_NAMES
        self.secondary_ilds_map = SECONDARY_ILDS_MAP

    def _init_sim_config(self, seed=None):
        return "/home/taekwon/projects/deeprl_signal_control/seoul/data/seoul.sumocfg"

    def plot_stat(self, rewards):
        self.state_stat['reward'] = rewards
        for name, data in self.state_stat.items():
            fig = plt.figure(figsize=(8, 6))
            plot_cdf(data)
            plt.ylabel(name)
            fig.savefig(self.output_path + self.name + '_' + name + '.png')


def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, color=c, label=label)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO)
    config = configparser.ConfigParser()
    config.read('./config/config_greedy_seoul.ini')
    base_dir = './output_result/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    env = SeoulEnv(config['ENV_CONFIG'], 2, base_dir, is_record=True, record_stat=True)
    ob = env.reset()
    controller = SeoulController(env.node_names, env.nodes)
    rewards = []
    it = 0
    while True:
        it += 1
        next_ob, _, done, reward = env.step(controller.forward(ob))
        rewards.append(reward)
        if done:
            break
        ob = next_ob
    # while True:
    #     next_ob, _, done, reward = env.baseCaseStep()
    #     rewards.append(reward)
    #     if done:
    #         break
    env.plot_stat(np.array(rewards))
    logging.info('avg reward: %.2f' % np.mean(rewards))
    env.terminate()
    time.sleep(2)
    env.collect_tripinfo()
    env.output_data()
