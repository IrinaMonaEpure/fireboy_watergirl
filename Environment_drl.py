from NoDisplayGame import NoDisplayGame
from NoDisplayBoard import NoDisplayBoard
from NoDisplayGates import NoDisplayGates
from NoDisplayDoors import NoDisplayFireDoor, NoDisplayWaterDoor
from NoDisplayCharacter import NoDisplayMagmaBoy, NoDisplayHydroGirl
from MagmaBoy_and_HydroGirl_Game.controller import ArrowsController, WASDController

import sys
import pygame
import numpy as np
import matplotlib.pyplot as plt
import time


class Event:
    def __init__(self, _type, key):
        self.type = _type
        self.key = key


class Environment_drl:
    def __init__(self, game, level):
        self.level = level
        self.state_vector_height = game.display_size[1]
        self.state_vector_width = game.display_size[0]
        self.states = np.zeros(
            (self.state_vector_width, self.state_vector_height), dtype=np.float32)
        # self.states = np.zeros((self.state_vector_height, self.state_vector_width), dtype=np.float32)
        self.n_states = (game.display_size[0] * game.display_size[1])
        self.action_list = [Event(pygame.KEYDOWN, pygame.K_LEFT), Event(pygame.KEYDOWN, pygame.K_RIGHT),
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
        self.n_actions = len(self.action_list)

    def update_states(self, rect_list, k):
        for rect_obj in rect_list:
            x_pos = rect_obj[1]
            y_pos = rect_obj[0]

            if len(rect_obj) == 4:
                w = rect_obj[3]
                h = rect_obj[2]

                if w > 0 and h > 0:
                    for i in range(w):
                        for j in range(h):
                            self.states[x_pos + i, y_pos + j] = k

                elif w < 0 and h < 0:
                    x_pos_new = x_pos - w
                    y_pos_new = y_pos - h
                    for i in range((-1) * w):
                        for j in range((-1) * h):
                            self.states[x_pos_new - i, y_pos_new - j] = k

            else:
                self.states[x_pos, y_pos] = k

    def perform_frame_update(self):
        # initialize empty matrix
        self.states = np.zeros(
            (self.state_vector_height, self.state_vector_width), dtype=np.float32)

        # environment objects
        lava_pools = self.board.get_lava_pools()
        self.update_states(lava_pools, 4)

        water_pools = self.board.get_water_pools()
        self.update_states(water_pools, 4)

        goo_pools = self.board.get_goo_pools()
        self.update_states(goo_pools, 5)

        f_door_location = [[self.fire_door_location[0],
                            self.fire_door_location[1], -16, -32]]
        self.update_states(f_door_location, 7)

        w_door_location = [[self.water_door_location[0],
                            self.water_door_location[1], -16, -32]]
        self.update_states(w_door_location, 8)

        magma_boy_loc = [self.magma_boy.rect]
        self.update_states(magma_boy_loc, 1)

        water_girl_loc = [self.hydro_girl.rect]
        self.update_states(water_girl_loc, 2)

        gate_location = self.gate.get_solid_blocks()
        self.update_states(gate_location, 10)

        #####
        solid_blocks = self.board.get_solid_blocks()
        self.update_states(solid_blocks, 6)

        ###
        plate_locations = self.gate.get_plates()
        self.update_states(plate_locations, 9)

    def reset(self):
        # load level data
        if self.level == "level1":
            self.board = NoDisplayBoard(
                'MagmaBoy_and_HydroGirl_Game/data/level1.txt')
            self.gate_location = (285, 128)
            self.plate_locations = [(190, 168), (390, 168)]
            self.gate = NoDisplayGates(
                self.gate_location, self.plate_locations)
            self.gates = [self.gate]

            self.fire_door_location = (64, 48)
            self.fire_door = NoDisplayFireDoor(self.fire_door_location)
            self.water_door_location = (128, 48)
            self.water_door = NoDisplayWaterDoor(self.water_door_location)
            self.doors = [self.fire_door, self.water_door]

            self.magma_boy_location = (16, 336)
            self.magma_boy = NoDisplayMagmaBoy(self.magma_boy_location)
            self.hydro_girl_location = (35, 336)
            self.hydro_girl = NoDisplayHydroGirl(self.hydro_girl_location)

        if self.level == "level2":
            self.board = NoDisplayBoard(
                'MagmaBoy_and_HydroGirl_Game/data/level2.txt')
            self.gates = []

            self.fire_door_location = (390, 48)
            self.fire_door = NoDisplayFireDoor(self.fire_door_location)
            self.water_door_location = (330, 48)
            self.water_door = NoDisplayWaterDoor(self.water_door_location)
            self.doors = [self.fire_door, self.water_door]

            self.magma_boy_location = (16, 336)
            self.magma_boy = NoDisplayMagmaBoy(self.magma_boy_location)
            self.hydro_girl_location = (35, 336)
            self.hydro_girl = NoDisplayHydroGirl(self.hydro_girl_location)

        if self.level == "level3":
            self.board = NoDisplayBoard(
                'MagmaBoy_and_HydroGirl_Game/data/level3.txt')
            self.gates = []

            self.fire_door_location = (5 * 16, 4 * 16)
            self.fire_door = NoDisplayFireDoor(self.fire_door_location)
            self.water_door_location = (28 * 16, 4 * 16)
            self.water_door = NoDisplayWaterDoor(self.water_door_location)
            self.doors = [self.fire_door, self.water_door]

            self.magma_boy_location = (28 * 16, 4 * 16)
            self.magma_boy = NoDisplayMagmaBoy(self.magma_boy_location)
            self.hydro_girl_location = (5 * 16, 4 * 16)
            self.hydro_girl = NoDisplayHydroGirl(self.hydro_girl_location)

        # get the frame 'pixels' (not actually pixels but placeholder values for them)
        self.perform_frame_update()

        return self.states

    def step(self, action, game):
        # initialize needed classes
        arrows_controller = ArrowsController()
        wasd_controller = WASDController()

        # move player
        arrows_controller.control_player([action], self.magma_boy)
        wasd_controller.control_player([action], self.hydro_girl)

        game.move_player(self.board, self.gates, [
                         self.magma_boy, self.hydro_girl])

        character_new_positions = [[self.magma_boy.rect.x, self.magma_boy.rect.y],
                                   [self.hydro_girl.rect.x, self.hydro_girl.rect.y]]

        # update frame
        self.perform_frame_update()

        s_next = self.states

        # check for player at special location
        game.check_for_death(self.board, [self.magma_boy, self.hydro_girl])

        game.check_for_gate_press(
            self.gates, [self.magma_boy, self.hydro_girl])

        game.check_for_door_open(self.fire_door, self.magma_boy)
        game.check_for_door_open(self.water_door, self.hydro_girl)

        done = False
        reward = 1

        if len(self.gates) > 0:
            for gate in self.gates:
                if gate.plate_is_pressed:
                    reward += 10

        if self.fire_door.player_at_door:
            reward += 10
        if self.water_door.player_at_door:
            reward += 10

        # special events
        if self.hydro_girl.is_dead() or self.magma_boy.is_dead():
            done = True
            reward = 0

        if game.level_is_done(self.doors):
            done = True
            reward = 100

        return s_next, reward, done

    def multistep(self, action, game):
        """Variant of step function that allows multiple actions to be performed at once."""

        # initialize needed classes
        arrows_controller = ArrowsController()
        wasd_controller = WASDController()

        # move player
        arrows_controller.control_player(action, self.magma_boy)
        wasd_controller.control_player(action, self.hydro_girl)

        game.move_player(self.board, self.gates, [
                         self.magma_boy, self.hydro_girl])

        character_new_positions = [[self.magma_boy.rect.x, self.magma_boy.rect.y],
                                   [self.hydro_girl.rect.x, self.hydro_girl.rect.y]]

        # update frame
        self.perform_frame_update()

        s_next = self.states

        # check for player at special location
        game.check_for_death(self.board, [self.magma_boy, self.hydro_girl])

        game.check_for_gate_press(
            self.gates, [self.magma_boy, self.hydro_girl])

        game.check_for_door_open(self.fire_door, self.magma_boy)
        game.check_for_door_open(self.water_door, self.hydro_girl)

        done = False
        reward = 1

        if len(self.gates) > 0:
            for gate in self.gates:
                if gate.plate_is_pressed:
                    reward += 10

        if self.fire_door.player_at_door:
            reward += 10
        if self.water_door.player_at_door:
            reward += 10

        # special events
        if self.hydro_girl.is_dead() or self.magma_boy.is_dead():
            done = True
            reward = 0

        if game.level_is_done(self.doors):
            done = True
            reward = 100

        return s_next, reward, done

    def copy(self, game):
        env_copy = Environment_drl(game, self.level)
        env_copy.reset()

        env_copy.gates = []
        if len(self.gates) > 0:
            for gate in self.gates:
                env_copy.gates.append(gate.copy())

        env_copy.doors = []
        for door in self.doors:
            env_copy.doors.append(door.copy())

        env_copy.magma_boy = self.magma_boy.copy(
            [self.magma_boy.rect.x, self.magma_boy.rect.y])
        env_copy.hydro_girl = self.hydro_girl.copy(
            [self.hydro_girl.rect.x, self.hydro_girl.rect.y])

        return env_copy


def experiment():
    game = NoDisplayGame()
    env = Environment_drl(game, "level1")
    s_0 = env.reset()

    frames = []
    for a in env.action_list:
        s_next, r, done = env.step(a, game)
        print("s_next: ", s_next.shape, " r: ", r, " done: ", done)
        frames.append(s_next)
        plt.imshow(s_next)
        plt.show()
        time.sleep(1)


if __name__ == "__main__":
    experiment()
