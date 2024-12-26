import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        self.actions = actions
        self.Ne = Ne 
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        state_tuple = tuple(state)
        self.N[state_tuple + (action,)] += 1

    def update_q(self, s, a, r, s_prime):
        max_q_next = 0 if s_prime is None else max(self.Q[tuple(s_prime) + (a_prime,)] for a_prime in self.actions)
        alpha = self.C / (self.C + self.N[tuple(s) + (a,)])
        self.Q[tuple(s) + (a,)] += alpha * (r + self.gamma * max_q_next - self.Q[tuple(s) + (a,)])

        
    def act(self, environment, points, dead):
        '''
        Act based on the current environment, updating Q and N if in training mode.
        '''
        i_Q = [0, 0, 0, 0, 0, 0, 0, 0]

        # Extract components from environment
        head_x, head_y = environment[0], environment[1]
        body = set(environment[2])  # Body positions as a set
        food_x, food_y = environment[3], environment[4]

        # Adjoining wall conditions
        if head_x == 40:  # Wall left of snake head
            i_Q[0] = 1
        elif head_x == 480:
            i_Q[0] = 2
        if head_y == 40:  # Wall above snake head
            i_Q[1] = 1
        elif head_y == 480:
            i_Q[1] = 2

        # Food position relative to snake head
        if head_x > food_x:
            i_Q[2] = 1  # Food left
        elif head_x < food_x:
            i_Q[2] = 2  # Food right
        if head_y > food_y:
            i_Q[3] = 1  # Food above
        elif head_y < food_y:
            i_Q[3] = 2  # Food below

        # Adjoining body segments
        if (head_x, head_y - 40) in body:  # Body above head
            i_Q[4] = 1
        if (head_x, head_y + 40) in body:  # Body below head
            i_Q[5] = 1
        if (head_x - 40, head_y) in body:  # Body left of head
            i_Q[6] = 1
        if (head_x + 40, head_y) in body:  # Body right of head
            i_Q[7] = 1

        if self._train:  # Training mode
            max_Q = -1000000
            max_a = 3
            reward = -0.1

            # Reward calculation
            if points - self.points == 1:
                reward = 1
            if dead:
                reward = -1

            if self.s is not None and self.a is not None:
                # Calculate max Q(s', a')
                for action in range(3, -1, -1):
                    if self.Q[i_Q[0], i_Q[1], i_Q[2], i_Q[3], i_Q[4], i_Q[5], i_Q[6], i_Q[7], action] > max_Q:
                        max_Q = self.Q[i_Q[0], i_Q[1], i_Q[2], i_Q[3], i_Q[4], i_Q[5], i_Q[6], i_Q[7], action]
                        max_a = action

                # Update N-table and calculate alpha
                self.N[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7], self.a] += 1
                alpha = self.C / (self.C + self.N[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7], self.a])

                # Update Q-table
                self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7], self.a] += \
                    alpha * (reward - self.Q[self.s[0], self.s[1], self.s[2], self.s[3], self.s[4], self.s[5], self.s[6], self.s[7], self.a] + self.gamma * max_Q)

            if dead:
                self.reset()
            else:
                # Save current state, points, and determine action
                self.s = i_Q
                self.points = points
                max_q_n = -1000000
                for action in range(3, -1, -1):
                    if self.N[i_Q[0], i_Q[1], i_Q[2], i_Q[3], i_Q[4], i_Q[5], i_Q[6], i_Q[7], action] < self.Ne:
                        self.a = action
                        break
                    if self.Q[i_Q[0], i_Q[1], i_Q[2], i_Q[3], i_Q[4], i_Q[5], i_Q[6], i_Q[7], action] > max_q_n:
                        self.a = action
                        max_q_n = self.Q[i_Q[0], i_Q[1], i_Q[2], i_Q[3], i_Q[4], i_Q[5], i_Q[6], i_Q[7], action]
        else:  # Evaluation mode
            max_Q = -1000000
            for action in range(3, -1, -1):
                if self.Q[i_Q[0], i_Q[1], i_Q[2], i_Q[3], i_Q[4], i_Q[5], i_Q[6], i_Q[7], action] > max_Q:
                    max_Q = self.Q[i_Q[0], i_Q[1], i_Q[2], i_Q[3], i_Q[4], i_Q[5], i_Q[6], i_Q[7], action]
                    self.a = action

        return self.a