import numpy as np
import tensorflow as tf
import numpy as np
from Environment_drl import Environment_drl
from NoDisplayGame import NoDisplayGame
from collections import deque
from tqdm import tqdm
import random
import pickle


# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class DRL_Agent():
    def __init__(self, env, states_shape, n_actions, learning_rate, epsilon, gamma, optimizer):
        self.env = env
        self.action_list = self.env.action_list
        self.states_shape = states_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.replay_memory = deque(maxlen=1000)

        self.optimizer = optimizer
        self.model = self.build_compile_model()
        self.target_model = self.build_compile_model()
        self.align_target_model()

    def build_compile_model(self):
        model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Conv2D(64, 5, strides=(3, 3),padding="valid", activation="relu",
        #                                  data_format="channels_first", input_shape = (400, 544, 1)))
        # model.add(tf.keras.layers.Conv2D(64, 4, strides=(2, 2),padding="valid", activation="relu",
        #                                  data_format="channels_first"))
        # model.add(tf.keras.layers.Conv2D(64, 3, strides=(1, 1),padding="valid", activation="relu",
        #                                  data_format="channels_first"))

        model.add(tf.keras.layers.Conv2D(64, 4, strides=(2, 2), padding='same', activation='relu',
                                         kernel_initializer='he_normal', input_shape=(400, 544, 1)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu'))
        model.add(tf.keras.layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(512, activation="relu", kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dense(256, activation="relu", kernel_initializer='he_uniform'))

        # model.add(tf.keras.layers.Dense(128, activation='relu'))
        # model.add(tf.keras.layers.Dense(64, activation='relu'))
        # model.add(tf.keras.layers.Dense(32, activation='relu'))

        model.add(tf.keras.layers.Dense(self.n_actions))

        model.compile(loss = tf.keras.losses.Huber(),
                      optimizer = self.optimizer,
                      metrics = ["accuracy"])

        # model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu',
        #                                  kernel_initializer='he_normal', input_shape=(400, 544, 1)))
        # model.add(tf.keras.layers.Dropout(0.1))
        # model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
        #                                  kernel_initializer='he_normal'))
        # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        #
        # model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
        #                                  kernel_initializer='he_normal'))
        # model.add(tf.keras.layers.Dropout(0.1))
        # model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',
        #                                  kernel_initializer='he_normal'))
        # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        #
        # model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',
        #                                  kernel_initializer='he_normal'))
        # model.add(tf.keras.layers.Dropout(0.1))
        # model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
        #                                  kernel_initializer='he_normal'))
        # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        #
        # model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Dense(512, activation='relu'))
        # model.add(tf.keras.layers.Dense(self.n_actions))
        #
        # model.compile(loss=tf.keras.losses.Huber(),
        #               optimizer= self.optimizer,
        #               metrics=["accuracy"])

        return model

    def align_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def update_memory(self, state, action,reward, state_next, done):
        self.replay_memory.append([state, action,reward, state_next, done])

    def select_action(self, frame):
        if np.random.rand() <= self.epsilon:
            action_index = np.random.randint(0, len(self.action_list))
            action = self.action_list[action_index]

            return action

        frame = np.expand_dims(np.asarray(frame).astype(np.float32), axis=0)
        q_values = self.model.predict(frame)

        return self.action_list[np.argmax(q_values[0])]


    def train_model(self, batch_size):
        memory_batch = random.sample(self.replay_memory, batch_size)

        states = np.zeros((batch_size,) + (400, 544, 1))
        next_states = np.zeros((batch_size,) + (400, 544, 1))
        action_indexes, rewards, done_list = [], [], []

        for i in range(len(memory_batch)):
            states[i] = memory_batch[i][0]
            a_index = self.action_list.index(memory_batch[i][1])
            action_indexes.append(a_index)
            rewards.append(memory_batch[i][2])
            next_states[i] = memory_batch[i][3]
            done_list.append(memory_batch[i][4])

        # perform batch prediction to save some speed
        target = self.model.predict(states)
        target_old = np.array(target)

        # predict next actions
        target_next = self.model.predict(next_states)

        # predict q-values
        target_val = self.target_model.predict(next_states)

        for i in range(len(memory_batch)):
            if done_list[i]:
                target[i][action_indexes[i]] = rewards[i]
            else:
                target[i][action_indexes[i]] = rewards[i] + self.gamma*(np.amax(target_next[i]))

        self.model.fit(states, target, batch_size = batch_size, verbose = 0)

        # current_states = [element[0] for element in memory_batch]
        # current_states_next = np.array([element[3] for element in memory_batch])
        #
        # q_val_from_current_states = self.model.predict(np.array(current_states))
        # q_val_from_current_states_next = self.target_model.predict(current_states_next)
        #
        # X_train = []
        # y_train = []
        #
        # for i, [state,action,reward,state_next,done] in enumerate(memory_batch):
        #     if not done:
        #         max_future_q = reward + self.gamma * np.max(q_val_from_current_states_next[i])
        #     else:
        #         max_future_q = reward
        #
        #     action_index = self.action_list.index(action)
        #     Q_s = q_val_from_current_states[i]
        #
        #     Q_s[action_index] = (1 - self.learning_rate) * Q_s[action_index] + self.learning_rate * max_future_q
        #
        #     X_train.append(state)
        #     y_train.append(Q_s)
        #
        # self.model.fit(np.array(X_train), np.array(y_train), batch_size = batch_size,
        #                verbose = 0, shuffle = True)

    def save_model(self,name):
        self.model.save(name)


def dqn_learning():
    game = NoDisplayGame()
    env = Environment_drl(game, "level1")

    states_shape = (400, 544)
    n_actions = env.n_actions
    learning_rate = 0.1
    epsilon = 0.1
    gamma = 0.7
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)
    batch_size = 64
    agent = DRL_Agent(env, states_shape, n_actions, learning_rate, epsilon, gamma, optimizer)
    target_update_counter = 0
    train_episodes = 25
    timesteps_per_episode = 1000

    reward_list = []
    frames_list = []

    for ep in tqdm(range(train_episodes)):
        state = env.reset()
        state = state.reshape(400, 544, 1)

        if ep >= 15:
            frames_list.append(state)

        done = False
        training_counter = 0

        for timestep in tqdm(range(timesteps_per_episode)):
            training_counter += 1
            action = agent.select_action(state)
            # print(f"action: {action}")

            state_next, reward, done = env.step(action, game)
            state_next = state_next.reshape(400, 544, 1)
            agent.update_memory(state, action, reward, state_next, done)
            reward_list.append(reward)

            if len(agent.replay_memory) > batch_size and training_counter == 10:
                print("Start Training ..... ")
                agent.train_model(batch_size)
                target_update_counter += 1
                training_counter = 0


            if target_update_counter == 10:
                agent.align_target_model()
                target_update_counter = 0

            state = state_next
            if ep >= 15:
                frames_list.append(state)

            if done:
                break

    print(reward_list)
    return frames_list

if __name__ == '__main__':
    print("Started running.......")
    frames_list = dqn_learning()

    file = open(r"D:\Facultate\Year 1\Semester 2\Modern Game AI\Assignment3/frames_list2.pckl", 'wb')
    pickle.dump(frames_list, file)
    file.close()





