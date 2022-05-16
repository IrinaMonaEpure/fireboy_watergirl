import numpy as np
from Environment import Environment
from NoDisplayGame import NoDisplayGame


def softmax(x, temp):
    ''' Computes the softmax of vector x with temperature parameter 'temp' '''
    x = x / temp # scale by temperature
    z = x - max(x) # substract max to prevent overflow of softmax 
    return np.exp(z)/np.sum(np.exp(z)) # compute softmax

def argmax(x):
    ''' Own variant of np.argmax with random tie breaking '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)
    

class MonteCarloAgent:
    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        # Initialize a table with means Q(s,a) to 0
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
                
            # TO DO: Add own code
            
            # Known best action
            a_best = argmax(self.Q_sa[s])
            
            prob = np.random.uniform(0,1)
            if prob < (1-epsilon): # with probability 1-epsilon, make a greedy choice
                a = a_best
            else: # with probability epsilon, randomly choose an action
                a_list = list(range(0, self.n_actions))
                a_list.pop(a_best)
                a = np.random.choice(a_list)
            
                
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
                
            # TO DO: Add own code
            
            # Compute the probability of each action
            pi_as = softmax(self.Q_sa[s], temp)
                
            # Generate a random number and choose the action based on that    
            prob = np.random.uniform(0,1)
            sum_probs = 0
            for b in range(self.n_actions):
                if (prob > sum_probs) and (prob <= (sum_probs + pi_as[b])):
                    a = b
                    break
                sum_probs += pi_as[b]
                
        return a
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        
        G_t = np.zeros(len(actions)+1)
        
        # Start reward sumation from 0
        for i in range(len(actions)-1, -1, -1):
            # Compute Monte Carlo target at each step
            G_t[i] = rewards[i] + self.gamma * G_t[i+1]
            
            # Update Q-table
            self.Q_sa[states[i]][actions[i]] = self.Q_sa[states[i]][actions[i]] + self.learning_rate * (G_t[i] - self.Q_sa[states[i]][actions[i]])
            
        pass
    

def monte_carlo(env, n_timesteps, max_episode_length, learning_rate, gamma, 
                policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    #env = StochasticWindyGridworld(initialize_model=False)
    # Q_value table is initialized
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []

    # TO DO: Write your Monte Carlo RL algorithm here!
    
    # n_timesteps is our budget
    while n_timesteps > 0:
        
        # Sample initial state
        s = env.reset()
        
        states = [s]
        actions = []
        rewards_update = []
        
        # Collect episode
        for i in range(max_episode_length):
            # Sample action (e-greedy or softmax)
            a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
            actions.append(a)
            
            # Simulate environment
            s_next, r, done = env.step(a)
            states.append(s_next)
            rewards.append(r)
            rewards_update.append(r)
            n_timesteps -= 1 # decrease remaining number of steps available
            s = s_next
            
            # If final state, end run
            if done:
                break
            
            if n_timesteps==0:
                break
            
        # Update Q_sa
        pi.update(states, actions, rewards_update)

    return rewards
    
    
def test():
    n_timesteps = 1000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'softmax' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    level = "level1"
    game = NoDisplayGame()
    env = Environment(game, level)

    rewards = monte_carlo(env, n_timesteps, max_episode_length, learning_rate,
                          gamma, policy, epsilon, temp)
    print("Obtained rewards: {}".format(rewards))
            
if __name__ == '__main__':
    test()