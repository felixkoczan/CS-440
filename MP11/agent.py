import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table. 
        state_tuple = tuple(state)
        self.N[state_tuple + (action,)] += 1
 

    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table. 
 
        if s_prime is None:  # Edge case for death
            max_q_next = 0
        else:
            max_q_next = max(self.Q[tuple(s_prime) + (a_prime,)] for a_prime in self.actions)
 
        alpha = self.C / (self.C + self.N[tuple(s) + (a,)])
    
    # Q-learning update formula
        self.Q[tuple(s) + (a,)] += alpha * (r + self.gamma * max_q_next - self.Q[tuple(s) + (a,)])
        

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        if dead:
            # Update Q for death scenario
            if self.s is not None and self.a is not None:
                reward = -1  # Death penalty
                self.update_q(self.s, self.a, reward, None)
            self.reset()
            return 0  # Arbitrary action after death

        # Determine reward
        reward = 1 if points > self.points else -0.1

        # Update Q and N if valid previous state-action
        if self.s is not None and self.a is not None:
            self.update_q(self.s, self.a, reward, s_prime)

        # Choose action
        if self._train:
            action = self.exploration_policy(s_prime)
        else:
            action = self.choose_greedy_action(s_prime)

        # Update agent's internal state
        self.s = s_prime
        self.a = action
        self.points = points

        return action

    def exploration_policy(self, state):
        # Exploration vs. exploitation
        state_tuple = tuple(state)
        q_values = [self.Q[state_tuple + (a,)] for a in self.actions]
        n_values = [self.N[state_tuple + (a,)] for a in self.actions]

        # Exploration: Check if any action is underexplored
        for action, n_value in enumerate(n_values):
            if n_value < self.Ne:
                return action  # Explore this action

        # Exploitation: Choose greedy action
        return self.choose_greedy_action(state)

    def choose_greedy_action(self, state):
        # Choose action with the highest Q-value
        state_tuple = tuple(state)
        q_values = [self.Q[state_tuple + (a,)] for a in self.actions]
        return np.argmax(q_values)
        

    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        # TODO - MP11: Implement this helper function that generates a state given an environment 
        head_x, head_y = environment[0], environment[1]
        body = set(environment[2]) 
        food_x, food_y = environment[3], environment[4]
        rock_x, rock_y = environment[5], environment[6]

        width, height = self.display_width, self.display_height

        walls = {(0, y) for y in range(height)} | {(width - 1, y) for y in range(height)} | \
                {(x, 0) for x in range(width)} | {(x, height - 1) for x in range(width)}

        if 0 <= rock_x < width and 0 <= rock_y < height:
            walls.add((rock_x, rock_y))
            walls.add((rock_x + 1, rock_y))
        out_of_bounds = not (0 <= head_x < width and 0 <= head_y < height)

        food_dir_x = 0 if head_x == food_x else (1 if food_x < head_x else 2)
        food_dir_y = 0 if head_y == food_y else (1 if food_y < head_y else 2)

        if out_of_bounds:
            adjoining_wall_x, adjoining_wall_y = 0, 0
        else:
            adjoining_wall_x = 1 if (head_x - 1, head_y) in walls else 2 if (head_x + 1, head_y) in walls else 0
            adjoining_wall_y = 1 if (head_x, head_y - 1) in walls else 2 if (head_x, head_y + 1) in walls else 0

        adjoining_body_top = int((head_x, head_y - 1) in body)
        adjoining_body_bottom = int((head_x, head_y + 1) in body)
        adjoining_body_left = int((head_x - 1, head_y) in body)
        adjoining_body_right = int((head_x + 1, head_y) in body)

        state = (
            food_dir_x, food_dir_y,
            adjoining_wall_x, adjoining_wall_y,
            adjoining_body_top, adjoining_body_bottom,
            adjoining_body_left, adjoining_body_right
        )
        return state