from MagmaBoy_and_HydroGirl_Game.doors import Doors
import pygame


class NoDisplayDoors(Doors):
    def load_images(self):
        """
        Load the images for the door
        """
        # load image of door frame and make transparent
        self.frame_image = pygame.image.load(
            "MagmaBoy_and_HydroGirl_Game/data/door_images/door_frame.png")
        self.frame_image.set_colorkey((255, 0, 255))
        # load image of background
        self.door_background = pygame.image.load(
            "MagmaBoy_and_HydroGirl_Game/data/door_images/door_background.png")
        
        
class NoDisplayFireDoor(NoDisplayDoors):
    def __init__(self, door_location):
        CHUNK_SIZE = 16
        # set door loaction as input door loaction
        self.door_location = door_location
        # set door background location as the same as the door
        self.background_location = door_location
        # since the frame is larger than the door, it has to be offset
        self.frame_location = (door_location[0] - CHUNK_SIZE, door_location[1]
                               - 2 * CHUNK_SIZE)
        # load unique door image
        self.door_image = pygame.image.load(
            "MagmaBoy_and_HydroGirl_Game/data/door_images/fire_door.png")
        super().__init__()


class NoDisplayWaterDoor(NoDisplayDoors):
    def __init__(self, door_location):
        CHUNK_SIZE = 16
        # set door loaction as input door loaction
        self.door_location = door_location
        # set door background location as the same as the door
        self.background_location = door_location
        # since the frame is larger than the door, it has to be offset
        self.frame_location = (door_location[0] - CHUNK_SIZE, door_location[1]
                               - 2 * CHUNK_SIZE)
        # load unique door image
        self.door_image = pygame.image.load(
            "MagmaBoy_and_HydroGirl_Game/data/door_images/water_door.png")
        super().__init__()