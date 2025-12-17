import numpy as np
import random
#import numpy as np: Imports the numpy library, a powerful package for numerical computing in Python, and aliases it as np.
#import random: Imports the random library for generating random numbers, which will be used for exploration in the reinforcement learning algorithm.


# Define environment dimensions and parameters
GRID_SIZE = 5
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
start_pos = (4, 0)   # Robot starting position (bottom-left corner)
diamond_pos = (4, 4) # Diamond position (bottom-right corner)
fire_positions = [(0, 1), (1, 0), (4, 2), (2, 3)]  # Fire positions as shown in the image

#GRID_SIZE = 5: Defines the grid as a 5x5 environment.
#ACTIONS: Specifies the possible actions the robot can take: moving UP, DOWN, LEFT, or RIGHT.
#start_pos = (4, 0): Sets the starting position of the robot at the bottom-left corner.
#diamond_pos = (4, 4): Sets the goal (diamond) position at the bottom-right corner.
#fire_positions: Lists the grid cells that contain fire, which the robot should avoid.



# Rewards and penalties
DIAMOND_REWARD = 10
FIRE_PENALTY = -10
MOVE_PENALTY = -1

#DIAMOND_REWARD: Reward for reaching the diamond.
#FIRE_PENALTY: Penalty for entering a cell with fire.
#MOVE_PENALTY: Penalty for each move, encouraging the agent to find the shortest path.



# Initialize Q-table
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

#q_table: Initializes the Q-table with zeros. It has dimensions (GRID_SIZE, GRID_SIZE, len(ACTIONS)),
#where each cell stores the Q-values for each action at that state (cell position).

#Q(s,a)=(1−α)×Q(s,a)+α×(R+γ×maxQ(s′,a′))

# Hyperparameters
alpha = 0.1   # Learning rate
gamma = 0.9   # Discount factor
epsilon = 0.9 # Exploration rate
episodes = 100

#alpha: The learning rate, which determines the extent to which newly acquired information overrides old information.
#gamma: The discount factor, which considers future rewards. A value closer to 1 values future rewards more.
#epsilon: The exploration rate, used for the epsilon-greedy strategy. High epsilon means more exploration.
#episodes: Number of training episodes to run.


# Helper functions
def get_reward(state):
    if state == diamond_pos:
        return DIAMOND_REWARD
    elif state in fire_positions:
        return FIRE_PENALTY
    else:
        return MOVE_PENALTY

#get_reward(state): Returns a reward based on the current state:
#Diamond position: DIAMOND_REWARD
#Fire position: FIRE_PENALTY
#Any other cell: MOVE_PENALTY


def get_next_state(state, action):
    x, y = state
    if action == "UP":
        x = max(0, x - 1)
    elif action == "DOWN":
        x = min(GRID_SIZE - 1, x + 1)
    elif action == "LEFT":
        y = max(0, y - 1)
    elif action == "RIGHT":
        y = min(GRID_SIZE - 1, y + 1)
    return (x, y)

#get_next_state(state, action): Calculates the next state based on the current state and action,
#ensuring the agent stays within grid boundaries.


# Q-learning algorithm
for episode in range(episodes):
    state = start_pos
    done = False
    
#for episode in range(episodes):: Loops through each episode to train the agent.
#state = start_pos: Sets the initial state (robot's position) for each episode.
#done = False: A flag to indicate if the episode has ended (robot reached the goal or fire).


    while not done:
        # Choose action (epsilon-greedy strategy)
        if random.uniform(0, 1) < epsilon:
            action = random.choice(ACTIONS)  # Explore
        else:
            action = ACTIONS[np.argmax(q_table[state[0], state[1]])]  # Exploit
#Epsilon-Greedy Strategy: The agent explores with probability epsilon and 
#exploits (chooses the best-known action) with probability 1 - epsilon.

        
        # Take action and observe reward
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)


#next_state: Calculates the next state based on the chosen action.
#reward: Gets the reward for the next_state.

        
        # Update Q-table
        action_index = ACTIONS.index(action)
        old_value = q_table[state[0], state[1], action_index]
        next_max = np.max(q_table[next_state[0], next_state[1]])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state[0], state[1], action_index] = new_value

#Action Index: Finds the index of the chosen action.
#Old Value: Retrieves the old Q-value for the current state-action pair.
#Next Max: Finds the max Q-value for the next state (best future reward).
#New Value: Updates the Q-value using the Q-learning formula:
#        Q(s,a)=(1−α)×Q(s,a)+α×(R+γ×maxQ(s′,a′))        
        
        
        # Transition to next state
        state = next_state

        # Check if reached goal or hit fire
        if state == diamond_pos or state in fire_positions:
            done = True

#state = next_state: Moves to the next_state.
#done = True: Ends the episode if the agent reaches the diamond or fire.


    # Decay epsilon for less exploration over time
    epsilon = max(0.1, epsilon * 0.99)

#Epsilon Decay: Reduces epsilon over time to encourage exploitation as the agent learns.



# Display learned Q-values for each state-action
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        print(f"State ({i}, {j}): {q_table[i, j]}")

#Displays Q-values for each action at each state, allowing us to understand which actions the agent prefers.


# Function to display the agent's path from start to goal
def display_path():
    state = start_pos
    path = [state]
    
    while state != diamond_pos:
        action_index = np.argmax(q_table[state[0], state[1]])
        action = ACTIONS[action_index]
        state = get_next_state(state, action)
        path.append(state)
        if state in fire_positions:  # Stop if it hits fire to avoid endless loop
            break

    print("Path taken by the agent:")
    print(path)

#display_path(): Shows the learned path from the start position to the goal (diamond).
#While Loop: Follows the highest Q-value actions at each state to reach the diamond.
#Break Condition: Stops if the agent encounters fire.

# Display the path
display_path()