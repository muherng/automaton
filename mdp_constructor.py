#file defines helper functions for constructing datasets for mdp
import numpy as np
import math
from random import randrange, choice
import itertools

#start with a toy set of transitions. 

def gen_data(self,params, length = 1): 
    #is there a way to do this fast with numpy?
    num_data = params['num_data']
    mdp_data = self.gen_mdp(num_data)

    return mdp_data
    #raise NotImplementedError

def gen_mdp(self,num_data):
    #generate an mdp 
    mdp_data = None
    for i in range(num_data): 
        # create new rewards 
        self.initialize_rewards()
        mdp_instance = self.print_mdp()
        #mdp_data[i,:] = mdp_instance
        #run value iteration
        trace_data = self.value_iteration()
        mdp_row = np.concatenate((mdp_instance,
                                        [self.eq_token],
                                        trace_data))
        if mdp_data is None:  
            mdp_data = np.zeros((num_data,np.size(mdp_row)))
        mdp_data[i,:] = mdp_row
    return mdp_data

#utilities for changing from grid representation to state representation
def grid_to_state(row,col): 
    raise NotImplementedError

def state_to_grid(state):
    raise NotImplementedError 

#initialize the rewards dictionary with default set to random
def initialize_rewards(self,max_reward=16,mode='random'):
    rows,cols = self.grid_size 
    rewards = {}
    for i in range(rows): 
        for j in range(cols):
            rewards[(i,j)] = np.random.randint(max_reward)
    self.rewards = rewards
    return rewards 

def gen_trace(self,transition_graph, input):
    raise NotImplementedError

def pretty_print(self,data): 
    row,col = np.shape(data)
    for i in range(row):
        pretty_row = [self.dic[data[i,j]] for j in range(col)]
        print(" ".join(pretty_row))

def print_mdp(self):
    "[state, reward, action, next_state] a total of 6 tokens plug the open/close bracket"
    mdp_data = np.zeros(self.states*10*4)
    flag = 0
    for i in range(self.grid_size[0]):
        for j in range(self.grid_size[1]):
            state_number = self.grid_size[1] * i + j
            state = (i, j)
            reward = self.rewards.get(state, 0)
            for action in self.actions:
                next_state = self.get_next_state(state, action)
                if next_state is None:
                    continue 
                next_state_number = self.grid_size[1] * next_state[0] + next_state[1]
                #print(f"[{state_number}, {reward}, {action}, {next_state_number}]")
                state_action = np.array([self.open_paren, 
                                         self.type_state,
                                         self.number_offset + state_number, 
                                         self.type_reward,
                                         self.number_offset + reward, 
                                         self.type_action,
                                         self.number_offset + action,
                                         self.type_state, 
                                         self.number_offset + next_state_number, 
                                         self.close_paren])
                #print('state_action: ', state_action)
                #print('flag: ', flag)
                mdp_data[flag:flag+10] = state_action
                flag = flag + 10
    return mdp_data[:flag]
# Define the grid world parameters   
def print_state(self,i,j,value=None):
    state_number = self.grid_size[1] * i + j
    state = (i, j)
    reward = self.rewards.get(state, 0)
    if not value:  
        value = self.state_values[state]
    neighbors = []
    for action in self.actions:
        next_state = self.get_next_state(state, action)
        if next_state is None:
            continue 
        if next_state != state:
            neighbor_number = self.grid_size[1] * next_state[0] + next_state[1]
            neighbors.append(neighbor_number + self.number_offset)
    first = np.array([self.open_paren,
              self.type_state,
              self.number_offset + state_number,
              self.type_reward, 
              self.number_offset + reward,
              self.type_value, 
              self.number_offset + value,
              self.type_neighbors])
    second = np.array(neighbors + [self.close_paren])
    return np.concatenate((first,second))

def print_trace(self,i,j):
    state_number = self.grid_size[1] * i + j
    state = (i, j)
    reward = self.rewards.get(state, 0)
    neighbor_values = []
    for action in self.actions:
        next_state = self.get_next_state(state, action)
        if next_state is None:
            continue 
        if next_state != state:
            neighbor_values.append(self.state_values[next_state])
    if neighbor_values:
        max_value = max(neighbor_values)
    else:
        max_value = 0
    gamma_max_value = np.floor(self.gamma * max_value)
    total_value = reward + gamma_max_value
    first = np.array([self.open_paren,
              self.type_state,
              self.number_offset + state_number,
              self.type_neighbor_values])
    offset_neighbor_values = [val+self.number_offset for val in neighbor_values]
    second = np.array(offset_neighbor_values + [self.max, 
                                         self.number_offset + max_value,
                                         self.type_discount, 
                                         self.number_offset + int(self.gamma*10),
                                         self.intermediate, 
                                         self.number_offset + gamma_max_value,
                                         self.type_reward, 
                                         self.number_offset + reward,
                                         self.final, 
                                         self.number_offset + total_value, 
                                         self.close_paren])
    return (np.concatenate((first,second)), total_value)
    
def get_next_state(self, state, action):
    if state in self.terminal_states:
        return state
    
    effect = self.action_effects[action]
    next_state = (state[0] + effect[0], state[1] + effect[1])
    
    if 0 <= next_state[0] < self.grid_size[0] and 0 <= next_state[1] < self.grid_size[1]:
        return next_state
    return None

def value_iteration(self):
    trace_data = None
    flag = 0
    for _ in range(self.iterations):
        new_state_values = np.copy(self.state_values)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                state_data =  self.print_state(i,j)
                #print('state_data: ', state_data)
                cot_data,new_value = self.print_trace(i,j)
                #print('cot_data: ', cot_data)
                new_state_data = self.print_state(i,j,new_value)
                #print('new_state_data: ', new_state_data)
                full_iter = np.concatenate((state_data,cot_data,new_state_data))
                if trace_data is None:
                    trace_data = np.zeros(np.size(full_iter)) 
                if flag >= np.size(trace_data): 
                    new_trace_data = np.zeros(2*np.size(trace_data))
                    new_trace_data[:flag] = trace_data
                    trace_data = new_trace_data
                trace_data[flag:flag+np.size(full_iter)] = full_iter  
                flag = flag + np.size(full_iter)
                state = (i, j)
                new_state_values[state] = new_value
        self.state_values = new_state_values
    return trace_data[:flag]

def get_policy(self):
    policy = {}
    for i in range(self.grid_size[0]):
        for j in range(self.grid_size[1]):
            state = (i, j)
            if state in self.terminal_states:
                policy[state] = None
            else:
                best_action = None
                best_value = float('-inf')
                for action in self.actions:
                    next_state = self.get_next_state(state, action)
                    if next_state is None:
                        continue
                    value = self.rewards.get(state, 0) + self.gamma * self.state_values[next_state]
                    if value > best_value:
                        best_value = value
                        best_action = action
                policy[state] = best_action
    return policy