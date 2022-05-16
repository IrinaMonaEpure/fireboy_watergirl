from MagmaBoy_and_HydroGirl_Game.board import Board


class NoDisplayBoard(Board):
    def __init__(self, path):
        self.CHUNK_SIZE = 16
        self.load_map(path)
        self.make_solid_blocks()
        self.make_water_pools()
        self.make_lava_pools()
        self.make_goo_pools()
        
    def load_images(self):
        pass