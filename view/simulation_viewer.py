import os
import numpy as np
import pygame

from model.car import Car
from model.track import Track
from view.map_previewer import MapPreviewer
from view.colors import TRACK_GREEN, TRACK_LINE, CAR_COLOR, SIDEBAR_BG, TEXT_COLOR, MAP_BG, ASTAR_LINE, BACK_BUTTON_COLOR


class SimulationViewer:
    def __init__(self, screen, font, width, height, map_area_width, clock):
        self.screen = screen
        self.font = font
        self.WIDTH = width
        self.HEIGHT = height
        self.map_area_width = map_area_width
        self.clock = clock

    def render_loop(self, track, car, best_line=None, grid_builder=None):
        map_area_width = self.map_area_width
        sidebar_width = self.WIDTH - map_area_width
        steps = 0
        start_ticks = pygame.time.get_ticks()
        car_finished = False
        summary_shown = False

        while True:
            self.screen.fill(MAP_BG)
            pygame.draw.rect(self.screen, SIDEBAR_BG, (map_area_width, 0, sidebar_width, self.HEIGHT))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            dx, dy = self._draw_track(track)

            if grid_builder:
                self.draw_grid(grid_builder)

            if best_line:
                pygame.draw.lines(self.screen, (0, 200, 255), False, list(zip(best_line[0], best_line[1])), 4)

            if not car_finished:
                car.move_car()
                if car.is_finished():
                    car_finished = True
                else:
                    car_position = car.get_position()
                    if car_position:
                        pygame.draw.circle(self.screen, CAR_COLOR, (int(car_position[0]), int(car_position[1])), 6)

            self._draw_sidebar(car, map_area_width)

            if car_finished and not summary_shown:
                real_time = (pygame.time.get_ticks() - start_ticks) / 1000.0
                print(
                    f"[{car.__class__.__name__}] Finished in {real_time:.2f} s, Max speed: {car.max_speed_reached * 3.6:.1f} km/h")
                self._draw_lap_summary(car, start_ticks, map_area_width)
                summary_shown = True

            pygame.display.flip()
            self.clock.tick(60)

            if car_finished and summary_shown:
                # Draw Replay button
                replay_btn_rect = pygame.Rect(map_area_width + 25, 380, sidebar_width - 50, 50)
                pygame.draw.rect(self.screen, (100, 100, 200), replay_btn_rect)
                replay_text = self.font.render("Replay", True, TEXT_COLOR)
                replay_text_rect = replay_text.get_rect(center=replay_btn_rect.center)
                self.screen.blit(replay_text, replay_text_rect)

                # Draw Back to Main Menu button
                back_btn_rect = pygame.Rect(map_area_width + 25, 630, sidebar_width - 50, 50)
                pygame.draw.rect(self.screen, BACK_BUTTON_COLOR, back_btn_rect)
                back_text = self.font.render("Back to Main Menu", True, TEXT_COLOR)
                back_text_rect = back_text.get_rect(center=back_btn_rect.center)
                self.screen.blit(back_text, back_text_rect)

                pygame.display.flip()

                while True:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            exit()
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if replay_btn_rect.collidepoint(event.pos):
                                return self._render_loop(track_x, track_y, Car(car.car_x, car.car_y, track_width),
                                                         track_width, best_line)
                            elif back_btn_rect.collidepoint(event.pos):
                                return  # Just exit back to the main menu

    def _draw_track(self, track):
        left_x, left_y, right_x, right_y = track.compute_boundaries()

        track_polygon = list(zip(left_x, left_y)) + list(zip(reversed(right_x), reversed(right_y)))
        pygame.draw.polygon(self.screen, TRACK_GREEN, track_polygon)

        behaviors = getattr(self, 'current_env', None)
        behaviors = behaviors.point_behaviors if behaviors else None

        if behaviors:
            for i in range(len(track.x) - 1):
                if behaviors[i] == "straight":
                    color = (220, 220, 220)
                else:
                    stage, _ = behaviors[i]
                    if stage == "entry":
                        color = (50, 100, 255)
                    elif stage == "apex":
                        color = (255, 50, 50)
                    elif stage == "exit":
                        color = (50, 255, 100)
                    elif stage == "adjust":
                        color = (250, 255, 100)
                    else:
                        color = (200, 200, 200)

                pygame.draw.line(self.screen, color,
                                 (track.x[i], track.y[i]),
                                 (track.x[i + 1], track.y[i + 1]),
                                 3)
        else:
            pygame.draw.lines(self.screen, TRACK_LINE, False, list(zip(track.x, track.y)), 2)

        return np.gradient(track.x), np.gradient(track.y)

    def _draw_sidebar(self, car, map_area_width):
        sidebar_title = self.font.render("Telemetry", True, TEXT_COLOR)
        self.screen.blit(sidebar_title, (map_area_width + 30, 40))
        pygame.draw.line(self.screen, (80, 80, 100), (map_area_width + 20, 65), (self.WIDTH - 20, 65), 2)

        speed_kph = car.speed * 3.6
        radius = car.compute_local_curvature_radius_by_distance()
        max_v = car.compute_max_lateral_speed(radius)
        radius_display = "-" if radius == float("inf") else f"{radius:.1f} m"
        max_v_display = f"{car.max_speed_reached * 3.6:.1f} km/h"

        telemetry_lines = [
            f"Speed: {speed_kph:.1f} km/h",
            f"Turn Radius: {radius_display}",
            f"Max V: {max_v_display}"
        ]
        for i, line in enumerate(telemetry_lines):
            text = self.font.render(line, True, TEXT_COLOR)
            self.screen.blit(text, (map_area_width + 40, 90 + i * 30))

    def _draw_lap_summary(self, car, start_ticks, map_area_width):
        sim_time = (pygame.time.get_ticks() - start_ticks) / 1000.0
        real_time_sec = sim_time
        distance_km = car.distance_traveled / 1000  # meters to km
        avg_speed_kph = (distance_km / real_time_sec) * 3600 if real_time_sec > 0 else 0

        summary_title = self.font.render("Lap Summary", True, TEXT_COLOR)
        self.screen.blit(summary_title, (map_area_width + 30, 220))
        pygame.draw.line(self.screen, (80, 80, 100), (map_area_width + 20, 245), (self.WIDTH - 20, 245), 2)

        summary_lines = [
            f"Time: {real_time_sec:.2f} s",
            f"Distance: {distance_km:.2f} km",
            f"Avg Speed: {avg_speed_kph:.1f} km/h"
        ]
        for i, line in enumerate(summary_lines):
            summary_text = self.font.render(line, True, TEXT_COLOR)
            self.screen.blit(summary_text, (map_area_width + 40, 260 + i * 30))
