from MagmaBoy_and_HydroGirl_Game.character import Character
import pygame


class NoDisplayMagmaBoy(Character):
    def __init__(self, location):
        self.image = pygame.image.load(
            'MagmaBoy_and_HydroGirl_Game/data/player_images/magmaboy.png')
        self.side_image = pygame.image.load(
            'MagmaBoy_and_HydroGirl_Game/data/player_images/magmaboyside.png')
        self._type = "magma"
        super().__init__(location)


class NoDisplayHydroGirl(Character):
    def __init__(self, location):
        self.image = pygame.image.load(
            'MagmaBoy_and_HydroGirl_Game/data/player_images/hydrogirl.png')
        self.side_image = pygame.image.load(
            'MagmaBoy_and_HydroGirl_Game/data/player_images/hydrogirlside.png')
        self._type = "water"
        super().__init__(location)