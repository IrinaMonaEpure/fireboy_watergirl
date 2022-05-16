from MagmaBoy_and_HydroGirl_Game.gates import Gates


class NoDisplayGates(Gates):
    def __init__(self, gate_location, plate_locations):
        # set initial locations and state of plates and gate
        self.gate_location = gate_location
        self.plate_locations = plate_locations
        self.plate_is_pressed = False
        self.make_rects()
        
    def load_images(self):
        pass