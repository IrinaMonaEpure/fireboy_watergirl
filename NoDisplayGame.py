from MagmaBoy_and_HydroGirl_Game.game import Game
import pygame


class NoDisplayGame(Game):
    "Similar to Game, but does not use a display."
    
    def __init__(self):
        """
        Initialize game.

        Create an internal display that only the game handles.
        """
        self.chunk_size = 16
        self.display_size = (34 * self.chunk_size, 25 * self.chunk_size)
        self.display = pygame.Surface(self.display_size)
        
    def draw_level_screen(self, level_select):
        pass
    
    def user_select_level(self, level_select, controller):
        pass
    
    def draw_level_select_indicator(self, level_select, level_index):
        pass
    
    def refresh_window(self):
        pass
    
    def adjust_scale(self):
        pass
    
    def draw_level_background(self, board):
        pass
    
    def draw_board(self, board):
        pass
    
    def draw_gates(self, gates):
        pass
    
    def draw_doors(self, doors):
        pass
    
    def draw_player(self, players):
        pass