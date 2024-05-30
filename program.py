import numpy as np
import math
from random import randrange, choice
import itertools
import torch
from program_constructor import *

class Program():
    def __init__(
        self,
        alphabet = 2,
        states = 2,
        word_length = 1,
        mode = 'random',
        preferred_dtype='int64',
        max_states = 100,
        max_alphabet = 10,
        max_word_length = 20
    ): 
        self.alphabet = alphabet
        self.states = states
        self.word_length = word_length
        #self.transitions = transitions
        self.mode=mode
        self.preferred_dtype = preferred_dtype

        base = self.alphabet + self.states

        self.padding_token = 0
        self.start_token = 1  # Before input
        self.eos_token = 2  # After target
        self.eq_token =  3  # the equal token after input (optional)
        self.open_paren =  4 
        self.close_paren =  5
        self.start_string =  6 
        self.end_string =  7
        #self.n_tokens = base + 8
        #n_tokens must not change over the course of model saving and loading.  
        #TODO: FIX 
        self.max_states = max_states
        self.max_alphabet = max_alphabet
        self.max_word_length = max_word_length
        self.n_tokens = self.max_states + self.max_alphabet + 10

        #special symbols relevant for creating mask in seq2seq models
        self.special_symbols = {'<pad>': self.padding_token,
                           '<bos>': self.start_token,
                           '<eos>': self.eos_token,
                           '<eq>': self.eq_token}
        #an arbitrary offset for the start of state tokens
        #the problem is that state will always increase 
        #for now  
        self.alphabet_offset = 10
        #we keep alphabet size at 2 for the most part,
        #we vary states and word_length for generalization 
        self.state_offset = self.alphabet_offset + self.alphabet  
        #this is only for pretty print 
        self.dic = {}
        for i in range(self.state_offset,self.state_offset + self.states): 
            self.dic[i] = 's' + str(i-self.state_offset)
        for i in range(self.alphabet_offset,self.alphabet_offset + self.alphabet): 
            self.dic[i] = 'a' + str(i-self.alphabet_offset)
        self.dic[self.start_token] = ""
        self.dic[self.open_paren] = '['
        self.dic[self.close_paren] = ']'
        self.dic[self.start_string] = '<'
        self.dic[self.end_string] = '>'
        self.dic[self.eq_token] = '='
        self.dic[self.padding_token] = ""
        self.dic[self.eos_token] = ""

    def generate_batch(self,bs, **kwargs):
        res = self._generate_batch(bs)
        return res

    def _generate_batch(self,bs):
        assert False, "Not implemented"

    @property
    def seq(self):
        assert False, "Not implemented"

class dfa(Program):
    def __init__(
        self,
        alphabet,
        states,
        word_length,
        mode,
        **kwargs,
    ):
        super().__init__(alphabet,states,word_length,mode,**kwargs)

    def _generate_batch(self,num_data):
        #TODO: create params
        params = {'num_data': num_data}
        data = self.gen_dfa(params)
        np.random.shuffle(data)
        data = data[:num_data,:]
        #print('data: ', data[:4,:])
        self.pretty_print(data[:4,:])
        return data 
    
    @property
    def seq(self): 
        #start, equal, end, 5*states*alphabet, 2 + word_length, 5*word_length
        #add 5 more tokens for good measure
        return 3 + 5*self.max_states*self.max_alphabet + 2 + 6*self.max_word_length + 5 
        #return 200
    
dfa.gen_dfa = gen_dfa
dfa.gen_automaton = gen_automaton
dfa.gen_input = gen_input
dfa.gen_trace = gen_trace
dfa.pretty_print = pretty_print 