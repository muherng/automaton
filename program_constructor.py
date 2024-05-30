#file defines helper functions for constructing datasets
import numpy as np
import math
from random import randrange, choice
import itertools

#states, transitions...
#generate random transitions? 
#start with a toy set of transitions. 

def gen_dfa(self,params, length = 1): 
    length = self.word_length
    #is there a way to do this fast with numpy?
    num_data = params['num_data']
    automaton_data, transition_graph = self.gen_automaton(num_data)
    input_data, input = self.gen_input(num_data,length) 
    trace_data = self.gen_trace(transition_graph,input)
    start = self.start_token*np.ones((num_data,1))
    equal = self.eq_token*np.ones((num_data,1))
    eos = self.eos_token*np.ones((num_data,1))
    data = np.concatenate([start,
                    automaton_data,
                    input_data,
                    equal,
                    trace_data,
                    eos
                    ],axis=1)
    return data.astype(int)
    #raise NotImplementedError
    
    #generate the automaton
    #generate the input
    #random input? Most input is not going to be accepted.  
    #most states are unlikely to be explored.  
    #states are represented  digits
    #alphabet represented by digits offset by number of states -- maybe binary for now? 

def gen_input(self,num_data,length):
    input = self.alphabet_offset + np.random.randint(0,self.alphabet,(num_data,length))
    input_data = np.concatenate([self.start_string*np.ones((num_data,1)) ,input, self.end_string*np.ones((num_data,1))], axis=1)
    return input_data, input

def gen_trace(self,transition_graph, input):
    num_data,length = np.shape(input)
    trace = np.zeros((num_data,5*length))
    #when indexing into the table recall alphabet is shifted by number of states
    #code design is easily buggy
    #TODO: fix the symbol, input, self.states offsets to be bug proof. NOW
    #Map from token to element of transition table 
    inverse_state_dic = {}
    for i in range(self.state_offset,self.state_offset+self.states): 
        inverse_state_dic[i] = i - self.state_offset
    inverse_alphabet_dic = {}
    for i in range(self.alphabet_offset,self.alphabet_offset+self.alphabet): 
        inverse_alphabet_dic[i] = i - self.alphabet_offset
    #input = input - self.states
    for graph in range(num_data):
        index = 0
        transition = transition_graph[:,:,graph]
        #curr_state and next_state are offset to zero 
        curr_state = 0
        for s in input[graph,:]:
            symbol = inverse_alphabet_dic[s]
            next_state = transition[curr_state,symbol]
            trace[graph,index:index+5] = [self.open_paren,
                                          curr_state + self.state_offset,
                                          symbol + self.alphabet_offset,
                                          next_state + self.state_offset,
                                          self.close_paren]
            #raise ValueError
            #[self.open_paren,curr_state,symbol,next_state,self.close_paren]
            index = index+5
            curr_state = next_state
    return trace

def gen_automaton(self,num_data):
    states = self.states
    alphabet = self.alphabet
    #start token 
    #state action next state triplet
    #create the random matrix first.  Then convert to triplets
    #the state and alphabet mapping may get more complex in the future.  
    #the dictionaries help us organize this complexity 
    state_dic = {}
    for i in range(states): 
        state_dic[i] = i + self.state_offset
    alphabet_dic = {}
    for i in range(alphabet): 
        alphabet_dic[i] = i+self.alphabet_offset
    transition_graph = np.random.randint(0,states,(states,alphabet,num_data))
    automaton = np.zeros((num_data,5*states*alphabet))
    for graph in range(num_data): 
        index = 0
        for s in range(states):
            for a in range(alphabet):  
                triplet = [self.open_paren,
                        state_dic[s], 
                        alphabet_dic[a],
                        state_dic[transition_graph[s,a,graph]],
                        self.close_paren]
                automaton[graph,index:index+5] = triplet
                index = index + 5

    return automaton, transition_graph

def pretty_print(self,data): 
    row,col = np.shape(data)
    for i in range(row):
        pretty_row = [self.dic[data[i,j]] for j in range(col)]
        print(" ".join(pretty_row))