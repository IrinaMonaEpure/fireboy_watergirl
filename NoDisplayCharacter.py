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
        
    def copy(self, location):
        magma_boy_copy = NoDisplayMagmaBoy(location)
        magma_boy_copy._alive = self._alive
        magma_boy_copy.moving_right = self.moving_right
        magma_boy_copy.moving_left = self.moving_left
        magma_boy_copy.jumping = self.jumping
        magma_boy_copy.y_velocity = self.y_velocity
        magma_boy_copy.air_timer = self.air_timer
        try:
            magma_boy_copy._movement = self._movement
        except:
            magma_boy_copy._movement = [0, 0]
        return magma_boy_copy


class NoDisplayHydroGirl(Character):
    def __init__(self, location):
        self.image = pygame.image.load(
            'MagmaBoy_and_HydroGirl_Game/data/player_images/hydrogirl.png')
        self.side_image = pygame.image.load(
            'MagmaBoy_and_HydroGirl_Game/data/player_images/hydrogirlside.png')
        self._type = "water"
        super().__init__(location)
        
    def copy(self, location):
        hydro_girl_copy = NoDisplayHydroGirl(location)
        hydro_girl_copy._alive = self._alive
        hydro_girl_copy.moving_right = self.moving_right
        hydro_girl_copy.moving_left = self.moving_left
        hydro_girl_copy.jumping = self.jumping
        hydro_girl_copy.y_velocity = self.y_velocity
        hydro_girl_copy.air_timer = self.air_timer
        try:
            hydro_girl_copy._movement = self._movement
        except:
            hydro_girl_copy._movement = [0, 0]
        return hydro_girl_copy