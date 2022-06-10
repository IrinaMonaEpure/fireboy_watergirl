import tensorflow as tf
import numpy as np
from Environment_drl import Environment_drl
from NoDisplayGame import NoDisplayGame
from collections import deque
from tqdm import tqdm
import random
import pickle
import pygame

class Event:
    def __init__(self, _type, key):
        self.type = _type
        self.key = key


class DRL_Agent():
    def __init__(self, env, states_shape, learning_rate, epsilon, gamma, optimizer):
        self.env = env
        self.action_list = self.env.action_list
        self.states_shape = states_shape
        self.action_sets = [
            # MOVE SETS FOR FIRST CHARACTER
            [Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_UP),Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT)],
            [Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_UP),Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT)],
            [Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(None, None), Event(None, None), Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYUP, pygame.K_LEFT)],
            [Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(None, None), Event(None, None), Event(pygame.KEYDOWN, pygame.K_RIGHT), Event(pygame.KEYUP, pygame.K_RIGHT)],

            # MOVE SETS FOR THE SECOND CHARACTER
            [Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_w), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYUP, pygame.K_a)],
            [Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_w), Event(pygame.KEYDOWN, pygame.K_w), Event(pygame.KEYDOWN, pygame.K_w), Event(pygame.KEYUP, pygame.K_d)],
            [Event(pygame.KEYDOWN, pygame.K_a), Event(pygame.KEYDOWN, pygame.K_a),Event(None, None), Event(None, None), Event(pygame.KEYUP, pygame.K_a), Event(pygame.KEYUP, pygame.K_a)],
            [Event(pygame.KEYDOWN, pygame.K_d), Event(pygame.KEYDOWN, pygame.K_d),Event(None, None), Event(None, None), Event(pygame.KEYUP, pygame.K_d), Event(pygame.KEYUP, pygame.K_d)],

            # 'DO NOTHING' MOVE SET
            [Event(None, None), Event(None, None), Event(None, None), Event(None, None), Event(None, None), Event(None, None)],

        ]
        self.n_action_sets = 9

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_memory = deque(maxlen=5000)

        self.optimizer = optimizer
        self.model = self.build_compile_model()
        self.target_model = self.build_compile_model()
        self.align_target_model()

    def build_compile_model(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(16, (4, 4), strides=4, padding='same', activation="relu", data_format='channels_last', input_shape=(400, 544, 6)))
        model.add(tf.keras.layers.Conv2D(16, (4, 4), strides=4, padding='same', activation="relu", data_format='channels_last'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), data_format='channels_last'))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation="relu", data_format='channels_last'))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", data_format='channels_last'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), data_format='channels_last'))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(self.n_action_sets, activation='softmax'))

        model.compile(loss = tf.keras.losses.Huber(),
                      optimizer = self.optimizer,
                      metrics = ["accuracy"])
        return model

    def align_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_memory(self, state_frames, action_preset, reward_avg, state_frames_next, done):
        self.replay_memory.append([state_frames, action_preset, reward_avg, state_frames_next, done])

    def select_action(self, frames):
        if np.random.rand() <= self.epsilon:
            print('==== take random action ====')
            action_index = np.random.randint(0, len(self.action_sets))
            actions = self.action_sets[action_index]

            return actions

        frames = np.expand_dims(np.asarray(frames).astype(np.float32), axis=0)
        q_values = self.model.predict(frames)

        print('==== take predicted action ====')
        return self.action_sets[np.argmax(q_values[0])]

    def train_model(self, batch_size):
        memory_batch = random.sample(self.replay_memory, batch_size)

        states_sets = np.zeros((batch_size,) + (400, 544, 6))
        next_states_sets = np.zeros((batch_size,) + (400, 544, 6))
        action_indexes, rewards, done_list = [], [], []

        for i in range(len(memory_batch)):
            states_sets[i] = memory_batch[i][0]
            a_index = self.action_sets.index(memory_batch[i][1])
            action_indexes.append(a_index)
            rewards.append(memory_batch[i][2])
            next_states_sets[i] = memory_batch[i][3]
            done_list.append(memory_batch[i][4])

        # perform batch prediction to save some speed
        target = self.model.predict(states_sets)
        target_old = np.array(target)

        # predict next actions
        target_next = self.model.predict(next_states_sets)

        # predict q-values
        target_val = self.target_model.predict(next_states_sets)

        for i in range(len(memory_batch)):
            if done_list[i]:
                target[i][action_indexes[i]] = rewards[i]
            else:
                target[i][action_indexes[i]] = rewards[i] + \
                    self.gamma*(np.amax(target_next[i]))

        self.model.fit(states_sets, target, batch_size = batch_size, verbose = 0)

    def save_model(self, name):
        self.model.save(name)


def dqn_learning():
    game = NoDisplayGame()

    # train on level 1 of the game
    env = Environment_drl(game, "level1")

    states_shape = (400, 544)
    learning_rate = 0.01
    epsilon = 0.1
    gamma = 0.9
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
    batch_size = 16
    agent = DRL_Agent(env, states_shape, learning_rate, epsilon, gamma, optimizer)
    target_update_counter = 0
    train_episodes = 15
    timesteps_per_episode = 250

    reward_per_ep = []
    frames_list = []
    state_frames = np.zeros((400,544, 6))

    for ep in tqdm(range(train_episodes)):
        state = env.reset()

        # the first 6 frames are identical
        state_frames[:,:,0] = state
        state_frames[:,:,1] = state
        state_frames[:,:,2] = state
        state_frames[:,:,3] = state
        state_frames[:,:,4] = state
        state_frames[:,:,5] = state

        done = False
        training_counter = 0

        if ep > 7:
            frames_list.append(state_frames)

        for _ in tqdm(range(timesteps_per_episode)):
            print('==== TRAINING ====')
            training_counter += 1
            action_preset = agent.select_action(state_frames)


            state_frames_next = np.zeros((400, 544, 6))
            reward_sum = 0
            reward_list = []

            # get the next 6 frames
            for i, action in enumerate(action_preset):
                state_next, reward, done = env.step(action, game)

                state_frames_next[:,:,i] = state_next
                reward_sum += reward


                if done:
                    # fill in the rest of the frames
                    for j in range(i,6):
                        state_frames_next[:,:,j] = state_next
                        reward_sum += reward

                    break
            reward_list.append(reward_sum)

            agent.update_memory(state_frames, action_preset, reward_sum, state_frames_next, done)


            if len(agent.replay_memory) > batch_size and training_counter == 10:
                print("Start Training ..... ")
                agent.train_model(batch_size)
                target_update_counter += 1
                training_counter = 0


            if target_update_counter == 20:
                agent.align_target_model()
                target_update_counter = 0

            state_frames = state_frames_next

            if ep > 7:
                frames_list.append(state_frames)

            if done:
                break

            reward_per_ep.append(reward_list)

    trained_weights = agent.model.get_weights()

    return [trained_weights,frames_list, reward_per_ep]


def dqn_testing(trained_weights):
    game = NoDisplayGame()
    env = Environment_drl(game, "level1")

    states_shape = (400, 544)
    learning_rate = 0.01
    epsilon = 0.1
    gamma = 0.9
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    agent = DRL_Agent(env, states_shape, learning_rate, epsilon, gamma, optimizer)
    test_episodes = 5
    timesteps_per_episode = 500

    frames_list = []
    reward_per_ep = []

    state_frames = np.zeros((400, 544, 6))

    agent.model.set_weights(trained_weights)

    for episode in tqdm(range(test_episodes)):
        state = env.reset()

        # the first 6 frames are identical
        state_frames[:, :, 0] = state
        state_frames[:, :, 1] = state
        state_frames[:, :, 2] = state
        state_frames[:, :, 3] = state
        state_frames[:, :, 4] = state
        state_frames[:, :, 5] = state

        done = False
        frames_list.append(state_frames)

        reward_list = []

        for _ in tqdm(range(timesteps_per_episode)):
            action_preset = agent.select_action(state_frames)

            state_frames_next = np.zeros((400, 544, 6))
            reward_sum = 0

            for i, action in enumerate(action_preset):
                state_next, reward, done = env.step(action, game)

                state_frames_next[:,:,i] = state_next
                reward_sum += reward

                reward_list.append(reward_sum)

                if done:
                    # fill in the rest of the frames
                    for j in range(i,6):
                        state_frames_next[:,:,j] = state_next
                        reward_sum += reward

                        reward_list.append(reward_sum)

                    break

            state_frames = state_frames_next
            frames_list.append(state_frames)

            if done:
                break

        reward_per_ep.append(reward_list)

    return [frames_list, reward_per_ep]



if __name__ == '__main__':
    [trained_weights,frames_list_training, reward_per_ep_training] = dqn_learning()

    # save the variables locally
    file = open(r"./trained_weights.pckl", 'wb')
    pickle.dump(trained_weights, file)
    file.close()

    file = open(r"./frames_list_training.pckl", 'wb')
    pickle.dump(frames_list_training, file)
    file.close()

    file = open(r"./reward_per_ep_training.pckl", 'wb')
    pickle.dump(reward_per_ep_training, file)
    file.close()

    [frames_list_testing, reward_per_ep_testing] = dqn_testing(trained_weights)

    # save the variables locally
    file = open(r"./frames_list_testing.pckl", 'wb')
    pickle.dump(frames_list_testing, file)
    file.close()

    file = open(r"./reward_per_ep_testing.pckl", 'wb')
    pickle.dump(reward_per_ep_testing, file)
    file.close()
