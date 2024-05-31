To generate random automaton and execution trace start with 
python generate_aut.py 
With defaults 2 states, word length 1

For a big loop over many states and word lengths run 
python generate_wrapper.py 
With defaults states from 1-20 and word length 1-20

For training with seq2seq tranformer of Vaswani et. al run
python train_aut.py 

For training with encoder only model run 
python train_aut.py --seq enc 

Consider trying 
python train_aut.py --seq enc --kind hybrid
python train_aut.py --seq enc --kind lstm 
python train_aut.py --seq enc --kind rnn

For training on one dataset at a time to build up to more states and word lengths run 
python train_wrapper.py 

Conda environment stored in environment.yaml

A reccomended way to run the code is 
python generate_aut.py 

Followed by 
python train_aut.py --seq enc

Models and validation/accuracy saved to folder saved_models/ 
Data is saved in folder data_aut/ 
