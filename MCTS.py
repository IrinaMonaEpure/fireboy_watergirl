import numpy as np
import pygame
from copy import deepcopy
#from Environment import Event, Environment
from Environment_drl import Event, Environment_drl
from NoDisplayGame import NoDisplayGame


class MCTSNode:
    def __init__(self, env, game, state, parent=None, parent_action=None, done=False):
        self.env = env
        self.game = game
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.done = done
        self.children = []
        self.N = 0  # number of visits
        self.Q = 0  # reward sum
        """
        self.untried_actions = [Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_RIGHT),
                                Event(pygame.KEYDOWN, pygame.K_UP), Event(pygame.KEYUP, pygame.K_LEFT),
                                Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_UP),
                                Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_d),
                                Event(pygame.KEYDOWN, pygame.K_w), Event(pygame.KEYUP, pygame.K_a),
                                Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYUP, pygame.K_w),
                                Event(None, None)]
        """
        self.untried_actions = [[Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT)],  # Fire Boy move left
                                [Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(
                                    pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT)],  # Fire Boy move right
                                [Event(pygame.KEYDOWN, pygame.K_UP), Event(pygame.KEYUP, pygame.K_UP), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(
                                    pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT)],  # Fire Boy jump left
                                [Event(pygame.KEYDOWN, pygame.K_UP), Event(pygame.KEYUP, pygame.K_UP), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(
                                    pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT)],  # Fire Boy jump right
                                [Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(
                                    pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a)],  # Water Girl move left
                                [Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(
                                    pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d)],  # Water Girl move right
                                [Event(pygame.KEYDOWN, pygame.K_w), Event(pygame.KEYUP, pygame.K_w), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(
                                    pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a)],  # Water Girl jump left
                                [Event(pygame.KEYDOWN, pygame.K_w), Event(pygame.KEYUP, pygame.K_w), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d)]]  # Water Girl jump right

    def expand(self):
        action = self.untried_actions.pop()
        _env = self.env.copy(self.game)
        s_next, reward, done = _env.multistep(action, self.game)
        child = MCTSNode(_env, self.game, s_next, parent=self,
                         parent_action=action, done=done)
        self.children.append(child)
        print("Expansion")
        return child

    def rollout(self):
        _env = self.env.copy(self.game)
        done = self.done
        total_reward = 0
        """
        possible_actions = [Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_RIGHT),
                            Event(pygame.KEYDOWN, pygame.K_UP), Event(
                                pygame.KEYUP, pygame.K_LEFT),
                            Event(pygame.KEYUP, pygame.K_RIGHT), Event(
                                pygame.KEYUP, pygame.K_UP),
                            Event(pygame.KEYDOWN, pygame.K_a), Event(
                                pygame.KEYDOWN, pygame.K_d),
                            Event(pygame.KEYDOWN, pygame.K_w), Event(
                                pygame.KEYUP, pygame.K_a),
                            Event(pygame.KEYUP, pygame.K_d), Event(
                                pygame.KEYUP, pygame.K_w),
                            Event(None, None)]
        
        possible_actions = [[Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT)],  # Fire Boy move left
                            [Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(
                                pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT)],  # Fire Boy move right
                            [Event(pygame.KEYDOWN, pygame.K_UP), Event(pygame.KEYUP, pygame.K_UP), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(
                                pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT)],  # Fire Boy jump left
                            [Event(pygame.KEYDOWN, pygame.K_UP), Event(pygame.KEYUP, pygame.K_UP), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(
                                pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT)],  # Fire Boy jump right
                            [Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(
                                pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a)],  # Water Girl move left
                            [Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(
                                pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d)],  # Water Girl move right
                            [Event(pygame.KEYDOWN, pygame.K_w), Event(pygame.KEYUP, pygame.K_w), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(
                                pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a)],  # Water Girl jump left
                            [Event(pygame.KEYDOWN, pygame.K_w), Event(pygame.KEYUP, pygame.K_w), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d)]]  # Water Girl jump right
        """
        possible_actions = [[Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT)],  # Fire Boy move left
                            [Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(
                                pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT)],  # Fire Boy move right
                            [Event(pygame.KEYDOWN, pygame.K_UP), Event(pygame.KEYUP, pygame.K_UP), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(
                                pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT)],  # Fire Boy jump left
                            [Event(pygame.KEYDOWN, pygame.K_UP), Event(pygame.KEYUP, pygame.K_UP), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(
                                pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT)],  # Fire Boy jump right
                            [Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(
                                pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a)],  # Water Girl move left
                            [Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(
                                pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d)],  # Water Girl move right
                            [Event(pygame.KEYDOWN, pygame.K_w), Event(pygame.KEYUP, pygame.K_w), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(
                                pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a)],  # Water Girl jump left
                            [Event(pygame.KEYDOWN, pygame.K_w), Event(pygame.KEYUP, pygame.K_w), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYUP, pygame.K_d)]]  # Water Girl jump right

        while not done:
            action = self.rollout_policy(possible_actions)
            s_next, reward, done = _env.multistep(action, self.game)
            total_reward += reward

        print("Total reward of rollout: ", total_reward)
        return total_reward

    def rollout_policy(self, possible_actions):
        # random policy
        action = possible_actions[np.random.randint(len(possible_actions))]
        print("Rollout policy chose action: ", action)
        return action

    def backpropagate(self, total_reward):
        self.N += 1
        self.Q += total_reward

        if self.parent:
            self.parent.backpropagate(total_reward)

        print("Backpropagation")

    def is_fully_expanded(self):
        if len(self.untried_actions) == 0:
            print("Is fully expanded")
            return True
        else:
            print("Is not fully expanded")
            return False

    def best_child(self, c=0.1):
        # Upper Confidence Bound
        scores = []
        for child in self.children:
            scores.append((child.Q / child.N) +
                          c * np.sqrt(2 * np.log(self.Q / self.N)))

        best_child = self.children[np.argmax(scores)]
        print("Best child: ", best_child)
        return best_child

    def tree_policy(self):
        node = self

        if not node.done:
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child()

        print("Tree policy returned: ", node)
        return node

    def best_action(self, num_simulations=100):
        for i in range(num_simulations):
            node = self.tree_policy()
            total_reward = node.rollout()
            node.backpropagate(total_reward)

        # set exploration to 0
        best_action = self.best_child(c=0)
        print("Best action: ", best_action)
        return best_action


def test():
    level = "level1"
    num_simulations = 1
    game = NoDisplayGame()
    env = Environment_drl(game, level)
    s = env.reset()
    node = MCTSNode(env, game, s)
    i = 0
    frame_list = []

    while not node.done:
        i += 1
        s = s.reshape(400, 544, 1)
        frame_list.append(s)
        node = node.best_action(num_simulations=num_simulations)
        a = node.parent_action
        s_next, r, done = env.multistep(a, game)
        s = s_next

    print("Game ended in ", i, "moves.")


if __name__ == '__main__':
    test()
