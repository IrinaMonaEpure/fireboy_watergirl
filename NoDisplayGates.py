from MagmaBoy_and_HydroGirl_Game.gates import Gates
import pygame


class NoDisplayGates(Gates):
    def __init__(self, gate_location, plate_locations):
        # set initial locations and state of plates and gate
        self.gate_location = gate_location
        self.plate_locations = plate_locations
        self.plate_is_pressed = False
        self._gate_is_open = False
        self.load_images()
        self.make_rects()
        
    def load_images(self):
        """
        Load images for gate and plates
        """
        # load gate image and make transparent
        self.gate_image = pygame.image.load('MagmaBoy_and_HydroGirl_Game/data/gates_and_plates/gate.png')
        self.gate_image.set_colorkey((255, 0, 255))
        # load plate image and make transparent
        self.plate_image = pygame.image.load('MagmaBoy_and_HydroGirl_Game/data/gates_and_plates/plate.png')
        self.plate_image.set_colorkey((255, 0, 255))