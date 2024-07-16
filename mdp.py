import numpy as np
import math
from random import randrange, choice
import itertools
import torch
from mdp_constructor import *

class mdp():
    def __init__(
        self,
        actions = 4,
        states = 16,
        iterations = 1,
        mode = 'random',
        preferred_dtype='int64',
        max_states = 100,
        max_number = 32
    ): 
        self.actions = actions
        self.states = states
        self.iterations = iterations
        self.max_number = max_number
        #self.transitions = transitions
        self.mode=mode
        self.preferred_dtype = preferred_dtype

        self.padding_token = 0
        self.start_token = 1  # Before input
        self.eos_token = 2  # After target
        self.eq_token =  3  # the equal token after input (optional)
        self.open_paren =  4 
        self.close_paren =  5
        self.start_comp =  6 
        self.end_comp =  7
        self.type_neighbors = 8
        self.type_neighbor_values = 9
        self.next_state = 10
        #variable types
        self.type_state = 11 
        self.type_action = 12
        self.type_reward = 13
        self.type_discount = 14
        self.type_value = 15

        self.max = 16
        self.intermediate = 17
        self.final = 18
        base=19
        #n_tokens must not change over the course of model saving and loading.  
        #TODO: FIX 
        self.max_states = max_states
        self.n_tokens = max(self.max_states,max_number) + base

        #special symbols relevant for creating mask in seq2seq models
        self.special_symbols = {'<pad>': self.padding_token,
                           '<bos>': self.start_token,
                           '<eos>': self.eos_token,
                           '<eq>': self.eq_token}
        #TODO: handle actions for policy extraction
        #states,actions,rewards,values all occupy same token space
        self.number_offset = base
        #this is only for pretty print 
        self.dic = {}
        for i in range(self.number_offset,self.number_offset + self.max_number): 
            self.dic[i] = str(i-self.number_offset)
        self.dic[self.start_token] = ""
        self.dic[self.open_paren] = '['
        self.dic[self.close_paren] = ']'
        self.dic[self.start_comp] = '<'
        self.dic[self.end_comp] = '>'
        self.dic[self.eq_token] = '='
        self.dic[self.padding_token] = ""
        self.dic[self.eos_token] = ""
        self.dic[self.type_action] = 'a'
        self.dic[self.type_state] = 's'
        self.dic[self.type_reward] = 'r'
        self.dic[self.type_value] = 'v'
        self.dic[self.max] = 'max: '
        self.dic[self.type_discount] = 'discount: '
        self.dic[self.intermediate] = 'intermediate: '
        self.dic[self.final] = 'output: '
        self.dic[self.type_neighbors] = 'neighbors: '
        self.dic[self.type_neighbor_values] = 'neighbor_values: '

    def generate_batch(self,bs, **kwargs):
        res = self._generate_batch(bs)
        return res

    def _generate_batch(self,bs):
        assert False, "Not implemented"

    @property
    def seq(self):
        assert False, "Not implemented"

class grid_world(mdp):
    def __init__(
        self,
        states,
        iterations,
        mode = 'random', 
        gamma=0.5,
        max_reward = 16,
        max_number = 32,
        **kwargs
    ):
        mode = 'random'
        super().__init__(states=states,iterations=iterations,mode=mode, max_number = max_number,**kwargs)
        self.grid_size = (np.sqrt(states).astype(int), np.sqrt(states).astype(int))
        self.terminal_states = []
        rewards,state_values = self.initialize(max_reward = max_reward)
        self.rewards = rewards
        self.state_values = state_values
        self.gamma = gamma
        self.actions = [0, 1, 2, 3]  # Corresponding to 'U', 'D', 'L', 'R'
        self.action_effects = {
            0: (-1, 0),  # 'U'
            1: (1, 0),   # 'D'
            2: (0, -1),  # 'L'
            3: (0, 1)    # 'R'
        }   

    def _generate_batch(self,num_data):
        #create params
        params = {'num_data': num_data}
        data = self.gen_data(params)
        np.random.shuffle(data)
        data = data[:num_data,:]
        self.pretty_print(data[:1,:])
        return data 
    
    @property
    def seq(self): 
        #start, equal, end, 5*states*alphabet, 2 + word_length, 5*word_length
        #add 5 more tokens for good measure
        return 200 
        #raise NotImplementedError
        #return 3 + 5*self.max_states*self.max_alphabet + 2 + 6*self.max_word_length + 5 
        #return 200
    
grid_world.gen_data = gen_data
grid_world.gen_mdp = gen_mdp
grid_world.gen_trace = gen_trace
grid_world.pretty_print = pretty_print 
grid_world.print_mdp = print_mdp
grid_world.print_trace = print_trace
grid_world.print_state = print_state 
grid_world.get_next_state = get_next_state
grid_world.value_iteration = value_iteration
grid_world.get_policy = get_policy
grid_world.initialize = initialize
