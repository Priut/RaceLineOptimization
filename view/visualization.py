import numpy as np
import pygame

from view.map_previewer import MapPreviewer
from view.demo_viewer import DemoViewer
from view.simulation_viewer import SimulationViewer
import view.utils as utils
from view.colors import TRACK_GREEN, TRACK_LINE, CAR_COLOR, SIDEBAR_BG, TEXT_COLOR, MAP_BG, ASTAR_LINE, \
    BACK_BUTTON_COLOR, BUTTON_COLOR, CONTRAST_TEXT_COLOR


class Visualizer:
    """
    Provides a graphical interface for displaying maps, running simulations, and showcasing demos in a racing simulator.
    Uses Pygame for rendering and integrates with helper classes for map preview, simulation, and demo playback.
    """
    def __init__(self, width=1300, height=700):
        """
        Initializes the Pygame window, fonts, and viewers (map preview, demo, simulation).
        Sets screen dimensions and layout parameters.
        """
        pygame.init()
        self.WIDTH, self.HEIGHT = width, height
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.SysFont('Consolas', 18)
        self.clock = pygame.time.Clock()
        self.map_area_width = 850
        self.mapPreviewer = MapPreviewer(self.screen, self.font, self.clock, self.WIDTH, self.HEIGHT)
        self.demoViewer = DemoViewer(self.screen, self.font, self.WIDTH, self.HEIGHT, self.map_area_width)
        self.simulationViewer = SimulationViewer(self.screen, self.font, self.WIDTH, self.HEIGHT, self.map_area_width, self.clock)

    def preview_map(self, track):
        """Displays a preview of a given track using the MapPreviewer."""
        self.mapPreviewer.preview_map(self, track)

    def _render_buttons_and_wait(self, title_text, button_labels):
        """
        Renders a centered title and a list of clickable buttons on the screen.
        Waits for user interaction and returns the selected action (as a string identifier).
        """
        self.screen.fill(MAP_BG)
        title = self.font.render(title_text, True, TEXT_COLOR)
        title_rect = title.get_rect(center=(self.WIDTH // 2, 100))
        self.screen.blit(title, title_rect)

        button_width, button_height, spacing = self.map_area_width - 450, 50, 80
        x_start = (self.WIDTH - button_width) // 2
        y_start = 200

        buttons = []
        for i, (label, color) in enumerate(button_labels):
            rect = pygame.Rect(x_start, y_start + i * spacing, button_width, button_height)
            pygame.draw.rect(self.screen, color, rect)
            text = self.font.render(label, True, CONTRAST_TEXT_COLOR)
            self.screen.blit(text, text.get_rect(center=rect.center))
            buttons.append((label.lower().replace(" ", "_"), rect))

        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for action, rect in buttons:
                        if rect.collidepoint(event.pos):
                            return action

    def show_main_menu(self):
        """Displays the main menu with buttons for: Generate Map, Simulate, Demo. Returns the selected option."""
        labels = [("Generate Map", BUTTON_COLOR),
                  ("Simulate", BUTTON_COLOR),
                  ("Demo", BUTTON_COLOR)]
        return self._render_buttons_and_wait("-- Racing Simulator --", labels)

    def show_simulation_menu(self):
        """Displays the simulation method menu with buttons for: Q-Learning, A*, Back to main menu. Returns the selected option."""
        labels = [("Q-Learning", BUTTON_COLOR),
                  ("A*", BUTTON_COLOR),
                  ("Back to main menu", BACK_BUTTON_COLOR)]
        return self._render_buttons_and_wait("-- Choose Simulation Method --", labels)

    def show_loading_screen(self, track, title):
        """Shows a loading screen with a scaled preview of the track and a centered title."""
        track.scale_map_to_left_area(self.map_area_width, self.HEIGHT)
        self.screen.fill(MAP_BG)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        title_text = self.font.render(title, True, TEXT_COLOR)
        self.screen.blit(title_text, title_text.get_rect(center=(self.WIDTH // 2, 50)))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        left_x, left_y, right_x, right_y = track.compute_boundaries()

        all_x = np.concatenate([left_x, right_x])
        all_y = np.concatenate([left_y, right_y])

        transform = utils.compute_scaling_and_transform(
            all_x, all_y,
            self.WIDTH * 0.8, self.HEIGHT * 0.7,
            self.WIDTH // 2, self.HEIGHT // 2 + 30
        )

        polygon = list(zip(left_x, left_y)) + list(zip(reversed(right_x), reversed(right_y)))
        pygame.draw.polygon(self.screen, TRACK_GREEN, [transform(px, py) for px, py in polygon])
        pygame.draw.lines(self.screen, TRACK_LINE, False, [transform(px, py) for px, py in zip(track.x, track.y)], 2)

        pygame.display.flip()
        pygame.time.wait(100)

    def render_q_agent(self, track, car, best_line):
        """Renders a simulation using a Q-learning agent. Delegates to SimulationViewer."""
        self.simulationViewer.render_loop(track, car, best_line=best_line)

    def render_astar(self, track, car, best_line):
        """Renders a simulation using an A* path planner."""
        self.simulationViewer.render_loop(track, car, best_line=best_line)

    def draw_grid(self, grid_builder):
        """Draws the simulation grid using cell rectangles provided by grid_builder. Colors differ based on cell type."""
        for rect, value in grid_builder.get_cell_rects():
            color = (40, 40, 40) if value == 0 else (150, 50, 50)
            pygame.draw.rect(self.screen, color, rect, 1)

    def show_demo_preview(self):
        """Displays the demo map selection screen via DemoViewer."""
        self.demoViewer.show_demo_preview()