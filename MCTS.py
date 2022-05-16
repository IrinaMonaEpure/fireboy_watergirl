import numpy as np
from pygame.locals import *
from copy import deepcopy
from Environment import Event, Environment
from NoDisplayGame import NoDisplayGame


class MCTSNode:
    def __init__(self, env, state, parent=None, parent_action=None, done=False):
        self.env = env
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.done = done
        self.children = []
        self.N = 0 # number of visits
        self.Q = 0 # reward sum
        self.untried_actions = [Event(pygame.KEYDOWN, K_LEFT), Event(pygame.KEYDOWN, K_RIGHT),
                                Event(pygame.KEYDOWN, K_UP), Event(pygame.KEYUP, K_LEFT),
                                Event(pygame.KEYUP, K_RIGHT), Event(pygame.KEYUP, K_UP),
                                Event(pygame.KEYDOWN, K_a), Event(pygame.KEYDOWN, K_d),
                                Event(pygame.KEYDOWN, K_w), Event(pygame.KEYUP, K_a),
                                Event(pygame.KEYUP, K_d), Event(pygame.KEYUP, K_w),
                                Event(None, None)]
        
    def expand(self):
        action = self.untried_actions.pop()
        _env = deepcopy(self.env)
        s_next, reward, done = _env.step(action)
        child = MCTSNode(_env, s_next, parent=self, parent_action=action, done=done)
        self.children.append(child)
        return child
    
    def rollout(self):
        _env = deepcopy(self.env)
        done = self.done
        total_reward = 0
        possible_actions = [Event(pygame.KEYDOWN, K_LEFT), Event(pygame.KEYDOWN, K_RIGHT),
                            Event(pygame.KEYDOWN, K_UP), Event(pygame.KEYUP, K_LEFT),
                            Event(pygame.KEYUP, K_RIGHT), Event(pygame.KEYUP, K_UP),
                            Event(pygame.KEYDOWN, K_a), Event(pygame.KEYDOWN, K_d),
                            Event(pygame.KEYDOWN, K_w), Event(pygame.KEYUP, K_a),
                            Event(pygame.KEYUP, K_d), Event(pygame.KEYUP, K_w),
                            Event(None, None)]

        while not done:
            action = self.rollout_policy(possible_actions)
            s_next, reward, done = _env.step(action)
            total_reward += reward
            
        return total_reward
    
    def rollout_policy(self, possible_actions):
        # random policy
        return possible_actions(np.random.randint(len(possible_actions)))
    
    def backpropagate(self, total_reward):
        self.N += 1
        self.Q += total_reward
        
        if self.parent:
            self.parent.backpropagate(total_reward)
            
    def is_fully_expanded(self):
        if len(self.untried_actions) == 0:
            return True
        else:
            return False
        
    def best_child(self, c=0.1):
        # Upper Confidence Bound
        scores = []
        for child in self.children:
            scores.append((child.Q / child.N) + np.sqrt(2 * np.log(self.Q / self.N)))
            
        return self.children[np.argmax(scores)]
    
    def tree_policy(self):
        node = self
        
        if not node.done:
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child()
        
        return node
    
    def best_action(self, num_simulations):
        for i in range(num_simulations):
            node = self.tree_policy()
            total_reward = node.rollout()
            node.backpropagate(total_reward)
            
        # set exploration to 0
        return self.best_child(c=0)
    
    
def test():
    level = "level1"
    game = NoDisplayGame()
    env = Environment(game, level)
    state = env.reset()
    node = MCTSNode(env, state)
    i = 0
    
    while not node.done:
        i += 1
        node = node.best_action()
        
    print("Game ended in ", i, "moves.")
    
    
if __name__ == '__main__':
    test()