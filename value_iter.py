import numpy as np

class state: 
    def __init__(self,actions,next_state):
        self.actions = actions
        self.next_state = next_state

# Define the grid world environment
class GridWorld:
    def __init__(self, grid_size, terminal_states, rewards, gamma=0.5):
        self.grid_size = grid_size
        self.terminal_states = terminal_states
        self.rewards = rewards
        self.gamma = gamma
        self.actions = [0, 1, 2, 3]  # Corresponding to 'U', 'D', 'L', 'R'
        self.action_effects = {
            0: (-1, 0),  # 'U'
            1: (1, 0),   # 'D'
            2: (0, -1),  # 'L'
            3: (0, 1)    # 'R'
        }
        self.state_values = np.zeros(grid_size)
        
    
    def print_mdp(self):
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                state_number = self.grid_size[1] * i + j
                state = (i, j)
                reward = self.rewards.get(state, 0)
                for action in self.actions:
                    next_state = self.get_next_state(state, action)
                    next_state_number = self.grid_size[1] * next_state[0] + next_state[1]
                    print(f"[{state_number}, {reward}, {action}, {next_state_number}]")

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
            if next_state != state:
                neighbor_number = self.grid_size[1] * next_state[0] + next_state[1]
                neighbors.append(neighbor_number)
        print(f"[{state_number}, {reward}, {value}, {neighbors}]")
        #np.array([{state_number}, {reward}, {value}, {neighbors}])

    def print_trace(self,i,j):
        state_number = self.grid_size[1] * i + j
        state = (i, j)
        reward = self.rewards.get(state, 0)
        neighbor_values = []
        for action in self.actions:
            next_state = self.get_next_state(state, action)
            if next_state == None:
                continue 
            if next_state != state:
                neighbor_values.append(self.state_values[next_state])
        if neighbor_values:
            max_value = max(neighbor_values)
        else:
            max_value = 0
        gamma_max_value = np.floor(self.gamma * max_value)
        total_value = reward + gamma_max_value
        print(f"[{state_number}, {neighbor_values}, {max_value}, {gamma_max_value}, {reward}, {total_value}]")
    


    def get_next_state(self, state, action):
        if state in self.terminal_states:
            return state
        
        effect = self.action_effects[action]
        next_state = (state[0] + effect[0], state[1] + effect[1])
        
        if 0 <= next_state[0] < self.grid_size[0] and 0 <= next_state[1] < self.grid_size[1]:
            return next_state
        return None

    def value_iteration(self, theta=0.0001):
        while True:
            delta = 0
            #print(self.state_values)
            new_state_values = np.copy(self.state_values)
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    print('Current State')
                    self.print_state(i,j)
                    print('Trace')
                    value = self.print_trace(i,j)
                    print('Updated State')
                    self.print_state(i,j,value)
                    state = (i, j)
                    if state in self.terminal_states:
                        continue
                    value = self.state_values[state]
                    new_value = max(
                        self.rewards.get(state, 0) + np.floor(self.gamma * self.state_values[self.get_next_state(state, action)])
                        for action in self.actions
                    )
                    new_state_values[state] = new_value
                    delta = max(delta, abs(value - new_value))
            self.state_values = new_state_values
            if delta < theta:
                break

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
                        value = self.rewards.get(state, 0) + self.gamma * self.state_values[next_state]
                        if value > best_value:
                            best_value = value
                            best_action = action
                    policy[state] = best_action
        return policy

# Define the grid world parameters
rows = 4
columns = 4
grid_size = (rows, columns)
terminal_states = []
rewards = {}
for i in range(rows): 
    for j in range(columns):
        rewards[(i,j)] = np.random.randint(16)

# Initialize the grid world and run value iteration
grid_world = GridWorld(grid_size, terminal_states, rewards)

print('PRINT MDP')
grid_world.print_mdp()

grid_world.value_iteration()

# Print the state values
print("State Values:")
print(grid_world.state_values)

# Print the optimal policy
policy = grid_world.get_policy()
print("\nOptimal Policy:")
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        print(policy[(i, j)], end=' ')
    print()
