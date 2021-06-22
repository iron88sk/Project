import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import configparser
import env
from envs.seoul_env import SeoulEnv, SeoulController
import numpy as np
import matplotlib

ilds_map ={'1_l', '1_r', '2_l', '2_r', '3_u', '3_d'}



class SeoulConuterEnv(SeoulEnv):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False, sumo_config=None):
        self.sumo_config = sumo_config
        self.ilds_map = ilds_map
        self.counts_map = dict()
        self.vehicles_in_lane = dict()
        for edge in set(self.ilds_map.values()):
             self.vehicles_in_lane[edge] = list()
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _init_sim_config(self, seed=None):
        return  self.sumo_config

    def step(self, action):
        self.count()
        super().step(action)

    def count(self):
        for ilds in self.ilds_map.keys():
            vid = self.sim.lanearea.getLastStepVehicleIDs(ild)


class TrafficCounter:
    def __init__(self, config, base_dir, sumo_config):
        self.config = config
        self.base_dir = base_dir
        self.sumo_config = sumo_config
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        self.env = SeoulCounterEnv(self.config['ENV_CONFIG'], 2,self.base_dir, is_record=True, record_stat=True, sumo_config=self.sumo_config)
        self.ob = env.reset()
        self.controller = SeoulController(self.env.node_names, self.env.nodes)
    
    def exploreGreedy(self):
            while True:
                it += 1
                next_ob, _, done, reward = self.env.step(self.controller.forward(self.ob))
                if done:
                    break
                self.ob = next_ob

            self.env.terminate()
            
        
    def run(self):
        self.exploreGreedy()

    

if __name__ == '__main__':
    







    config = configparser.ConfigParser()
    config.read('./config/config_greedy_seoul.ini')
    base_dir = './output_result/'
    counter = TrafficCounter(config, base_dir)
    counter.run()
