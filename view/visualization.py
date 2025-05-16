import os

import numpy as np
import pygame

from model.car import Car
from model.track import Track
from view.map_previewer import MapPreviewer
from view.demo_viewer import DemoViewer
from view.simulation_viewer import SimulationViewer
import view.utils as utils
from view.colors import TRACK_GREEN, TRACK_LINE, CAR_COLOR, SIDEBAR_BG, TEXT_COLOR, MAP_BG, ASTAR_LINE, BACK_BUTTON_COLOR


class Visualizer:
    def __init__(self, width=1300, height=700):
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
        self.mapPreviewer.preview_map(self, track)

    def _render_buttons_and_wait(self, title_text, button_labels):
        self.screen.fill((30, 30, 30))
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
            text = self.font.render(label, True, TEXT_COLOR)
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
        labels = [("Generate Map", (100, 100, 200)),
                  ("Simulate", (100, 100, 200)),
                  ("Demo", (100, 100, 200))]
        return self._render_buttons_and_wait("Racing AI Simulator", labels)

    def show_simulation_menu(self):
        labels = [("QLearning Agent", (100, 100, 200)),
                  ("A*", (100, 100, 200)),
                  ("Back to main menu", BACK_BUTTON_COLOR)]
        return self._render_buttons_and_wait("Choose Simulation Method", labels)


    def show_loading_screen(self, track, title):
        track.scale_map_to_left_area(self.map_area_width, self.HEIGHT)
        self.screen.fill((30, 30, 30))
        title_text = self.font.render(title, True, TEXT_COLOR)
        self.screen.blit(title_text, title_text.get_rect(center=(self.WIDTH // 2, 50)))

        left_x, left_y, right_x, right_y = track.compute_boundaries()
        print(track.width)

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

    def render_q_agent(self, track, car, best_line):
        self.simulationViewer.render_loop(track, car, best_line=best_line)

    def render_astar(self, track, car, best_line, grid_builder=None):
        self.simulationViewer.render_loop(track, car, best_line=best_line, grid_builder=grid_builder)

    def draw_grid(self, grid_builder):
        for rect, value in grid_builder.get_cell_rects():
            color = (40, 40, 40) if value == 0 else (150, 50, 50)
            pygame.draw.rect(self.screen, color, rect, 1)

    def show_demo_preview(self):
        self.demoViewer.show_demo_preview()