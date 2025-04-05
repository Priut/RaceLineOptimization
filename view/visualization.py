import numpy as np
import pygame
from view.map_previewer import MapPreviewer
from view.colors import TRACK_GREEN, TRACK_LINE, CAR_COLOR, SIDEBAR_BG, TEXT_COLOR


class Visualizer:
    def __init__(self, width=1300, height=700):
        pygame.init()
        self.WIDTH, self.HEIGHT = width, height
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.SysFont('Consolas', 18)
        self.clock = pygame.time.Clock()
        self.time_scale = 10  # Simulation runs 10x faster
        self.unit_length_km = 5.0 / 1000  # 5 km / 1000 steps = 0.005 km/step

    def preview_map(self, x, y, track_width):
        previewer = MapPreviewer(self.screen, self.font, self.clock, self.WIDTH, self.HEIGHT)
        previewer.preview_map(self, x, y, track_width)

    def show_main_menu(self):
        self.screen.fill((30, 30, 30))

        title = self.font.render("🏎️ Racing AI Simulator", True, (255, 255, 255))
        title_rect = title.get_rect(center=(self.WIDTH // 2, 100))
        self.screen.blit(title, title_rect)

        # Define buttons
        buttons = [
            {"label": "Generate Map", "rect": pygame.Rect(self.WIDTH // 2 - 100, 250, 200, 50)},
            {"label": "Simulate", "rect": pygame.Rect(self.WIDTH // 2 - 100, 330, 200, 50)}
        ]

        for btn in buttons:
            pygame.draw.rect(self.screen, (100, 100, 200), btn["rect"])
            label = self.font.render(btn["label"], True, (255, 255, 255))
            label_rect = label.get_rect(center=btn["rect"].center)
            self.screen.blit(label, label_rect)

        pygame.display.flip()

        # Wait for click
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    for btn in buttons:
                        if btn["rect"].collidepoint(pos):
                            return btn["label"].lower().replace(" ", "_")  # returns "generate_map" or "simulate"


    def scale_points(self, x, y, margin=50):
        x = (x - min(x)) / (max(x) - min(x)) * (self.WIDTH - 2 * margin) + margin
        y = (y - min(y)) / (max(y) - min(y)) * (self.HEIGHT - 2 * margin) + margin
        return x, y

    def show_training_screen(self):
        self.screen.fill((30, 30, 30))
        message = self.font.render("Training AI... Please wait", True, (255, 255, 255))
        rect = message.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(message, rect)
        pygame.display.flip()

    def show_race_summary(self, sim_time_sec, steps):
        distance_km = steps * self.unit_length_km
        real_time_sec = sim_time_sec * self.time_scale  # Corrected: simulation was sped up
        avg_speed_kph = (distance_km / real_time_sec) * 3600 if real_time_sec > 0 else 0

        self.screen.fill((30, 30, 30))
        summary_lines = [
            "🏁 Race Summary 🏁",
            f"Time: {real_time_sec:.2f} s",
            f"Distance: {distance_km:.2f} km",
            f"Avg Speed: {avg_speed_kph:.1f} km/h"
        ]
        for i, line in enumerate(summary_lines):
            text = self.font.render(line, True, (255, 255, 255))
            rect = text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 60 + i * 40))
            self.screen.blit(text, rect)

        pygame.display.flip()

        # Wait for quit or key press
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False

    def draw_car_telemetry(self, car):
        bar_height = 150
        bar_width = 20
        throttle_y = self.HEIGHT - 200
        throttle_level = int(car.throttle * bar_height)
        pygame.draw.rect(self.screen, (100, 100, 100), (self.WIDTH - 40, throttle_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, (0, 200, 0),
                         (self.WIDTH - 40, throttle_y + bar_height - throttle_level, bar_width, throttle_level))

        speed_kph = car.speed * 3.6
        radius = car.compute_local_curvature_radius(car.car_index)
        max_v = car.compute_max_lateral_speed(radius)

        info = [
            f"Speed: {speed_kph:.1f} km/h",
            f"Turn Radius: {radius:.1f} m",
            f"Max V: {max_v:.1f} m/s"
        ]
        for i, text in enumerate(info):
            rendered = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(rendered, (20, 20 + i * 25))

    def render(self, track_x, track_y, car, track_width, best_line=None):
        dx = np.gradient(track_x)
        dy = np.gradient(track_y)
        length = np.sqrt(dx ** 2 + dy ** 2)
        length[length == 0] = 1
        dx /= length
        dy /= length

        left_x = track_x + (track_width / 2) * dy
        left_y = track_y - (track_width / 2) * dx
        right_x = track_x - (track_width / 2) * dy
        right_y = track_y + (track_width / 2) * dx

        running = True
        steps = 0
        start_ticks = pygame.time.get_ticks()

        while running:
            self.screen.fill((46, 46, 46))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            track_polygon = list(zip(left_x, left_y)) + list(zip(reversed(right_x), reversed(right_y)))
            pygame.draw.polygon(self.screen, (55, 158, 74), track_polygon)
            pygame.draw.lines(self.screen, (255, 255, 255), False, list(zip(track_x, track_y)), 2)

            car.move_car()
            car_position = car.get_position()
            if car_position:
                self.draw_car_telemetry(car)
                danger = car.speed > car.compute_max_lateral_speed(
                    car.compute_local_curvature_radius(car.car_index))
                color = (255, 50, 50) if danger else (255, 255, 0)
                pygame.draw.circle(self.screen, color, (int(car_position[0]), int(car_position[1])), 6)
                steps += 1
            else:
                running = False

            if best_line:
                pygame.draw.lines(self.screen, (0, 200, 255), False, list(zip(best_line[0], best_line[1])), 2)

            pygame.display.flip()
            self.clock.tick(60)

        sim_time = (pygame.time.get_ticks() - start_ticks) / 1000.0
        self.show_race_summary(sim_time, steps)
        pygame.quit()
