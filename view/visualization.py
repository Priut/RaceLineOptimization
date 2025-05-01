import os

import numpy as np
import pygame

from model.car import Car
from view.map_previewer import MapPreviewer
from view.colors import TRACK_GREEN, TRACK_LINE, CAR_COLOR, SIDEBAR_BG, TEXT_COLOR, MAP_BG, ASTAR_LINE, BACK_BUTTON_COLOR


class Visualizer:
    def __init__(self, width=1300, height=700):
        pygame.init()
        self.WIDTH, self.HEIGHT = width, height
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.SysFont('Consolas', 18)
        self.clock = pygame.time.Clock()
        self.map_area_width = 850

    def preview_map(self, x, y, track_width):
        MapPreviewer(self.screen, self.font, self.clock, self.WIDTH, self.HEIGHT).preview_map(self, x, y, track_width)

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


    def scale_map_to_left_area(self, x, y, margin=50):
        drawable_width = self.map_area_width - 2 * margin
        drawable_height = self.HEIGHT - 2 * margin
        x = (x - np.min(x)) / (np.max(x) - np.min(x)) * drawable_width + margin
        y = (y - np.min(y)) / (np.max(y) - np.min(y)) * drawable_height + margin
        x += margin // 2
        return x, y

    def _compute_track_edges(self, x, y, width):
        dx = np.gradient(x)
        dy = np.gradient(y)
        length = np.hypot(dx, dy)
        length[length == 0] = 1
        dx /= length
        dy /= length

        left_x = x + (width / 2) * dy
        left_y = y - (width / 2) * dx
        right_x = x - (width / 2) * dy
        right_y = y + (width / 2) * dx
        return left_x, left_y, right_x, right_y

    def _compute_scaling_and_transform(self, all_x, all_y, target_width, target_height, center_x, center_y):
        scale_x = target_width / (np.max(all_x) - np.min(all_x) + 1e-5)
        scale_y = target_height / (np.max(all_y) - np.min(all_y) + 1e-5)
        scale = min(scale_x, scale_y)
        offset_x = center_x - ((np.min(all_x) + np.max(all_x)) / 2) * scale
        offset_y = center_y - ((np.min(all_y) + np.max(all_y)) / 2) * scale

        def transform(px, py):
            return int(px * scale + offset_x), int(py * scale + offset_y)

        return transform

    def show_loading_screen(self, x, y, track_width, title):
        self.screen.fill((30, 30, 30))
        title_text = self.font.render(title, True, TEXT_COLOR)
        self.screen.blit(title_text, title_text.get_rect(center=(self.WIDTH // 2, 50)))

        left_x, left_y, right_x, right_y = self._compute_track_edges(x, y, track_width)

        all_x = np.concatenate([left_x, right_x])
        all_y = np.concatenate([left_y, right_y])

        transform = self._compute_scaling_and_transform(
            all_x, all_y,
            self.WIDTH * 0.8, self.HEIGHT * 0.7,
            self.WIDTH // 2, self.HEIGHT // 2 + 30
        )

        polygon = list(zip(left_x, left_y)) + list(zip(reversed(right_x), reversed(right_y)))
        pygame.draw.polygon(self.screen, TRACK_GREEN, [transform(px, py) for px, py in polygon])
        pygame.draw.lines(self.screen, TRACK_LINE, False, [transform(px, py) for px, py in zip(x, y)], 2)
        pygame.display.flip()

    def _draw_track(self, track_x, track_y, track_width):
        left_x, left_y, right_x, right_y = self._compute_track_edges(track_x, track_y, track_width)

        track_polygon = list(zip(left_x, left_y)) + list(zip(reversed(right_x), reversed(right_y)))
        pygame.draw.polygon(self.screen, TRACK_GREEN, track_polygon)

        behaviors = getattr(self, 'current_env', None)
        behaviors = behaviors.point_behaviors if behaviors else None

        if behaviors:
            for i in range(len(track_x) - 1):
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
                                 (track_x[i], track_y[i]),
                                 (track_x[i + 1], track_y[i + 1]),
                                 3)
        else:
            pygame.draw.lines(self.screen, TRACK_LINE, False, list(zip(track_x, track_y)), 2)

        return np.gradient(track_x), np.gradient(track_y)

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

    def render_q_agent(self, track_x, track_y, car, track_width, best_line):
        self._render_loop(track_x, track_y, car, track_width, best_line=best_line)

    def render_astar(self, track_x, track_y, car, track_width, best_line, grid_builder=None):
        self._render_loop(track_x, track_y, car, track_width, best_line=best_line, grid_builder=grid_builder)

    def _render_loop(self, track_x, track_y, car, track_width, best_line=None, grid_builder=None):
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

            dx, dy = self._draw_track(track_x, track_y, track_width)

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


    def draw_grid(self, grid_builder):
        for rect, value in grid_builder.get_cell_rects():
            color = (40, 40, 40) if value == 0 else (150, 50, 50)
            pygame.draw.rect(self.screen, color, rect, 1)

    def display_map_grid(self, map_data_list, filenames):
        self.screen.fill((30, 30, 30))
        grid_rows, grid_cols = 2, 2
        cell_w = self.WIDTH // grid_cols
        cell_h = self.HEIGHT // grid_rows
        margin = 30
        map_rects = []

        back_icon_path = os.path.join("..", "view", "icons", "previous.png")
        back_icon = pygame.image.load(back_icon_path)
        back_icon = pygame.transform.scale(back_icon, (40, 40))
        back_rect = back_icon.get_rect(topleft=(20, 20))

        for idx, (x, y, width) in enumerate(map_data_list):
            row, col = divmod(idx, grid_cols)
            offset_x = col * cell_w
            offset_y = row * cell_h

            left_x, left_y, right_x, right_y = self._compute_track_edges(x, y, width)

            all_x = np.concatenate([left_x, right_x])
            all_y = np.concatenate([left_y, right_y])
            transform = self._compute_scaling_and_transform(
                all_x, all_y,
                cell_w - 2 * margin, cell_h - 2 * margin,
                offset_x + cell_w // 2, offset_y + cell_h // 2
            )

            polygon = list(zip(left_x, left_y)) + list(zip(reversed(right_x), reversed(right_y)))
            pygame.draw.polygon(self.screen, TRACK_GREEN, [transform(px, py) for px, py in polygon])
            pygame.draw.lines(self.screen, TRACK_LINE, False, [transform(px, py) for px, py in zip(x, y)], 2)
            map_rects.append(pygame.Rect(offset_x, offset_y, cell_w, cell_h))

        self.screen.blit(back_icon, back_rect)
        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if back_rect.collidepoint(event.pos):
                        return None
                    for i, rect in enumerate(map_rects):
                        if rect.collidepoint(event.pos):
                            return filenames[i]

    def run_demo_with_paths(self, map_filename):
        base_directory= "/home/priut/Documents/disertatie/RaceLineOptimization"
        map_path = map_filename  # already full path now
        base_name = os.path.splitext(os.path.basename(map_filename))[0]
        path_dir = os.path.join(base_directory, "demo_maps", "paths")
        q_path_file = os.path.join(path_dir, f"{base_name}_q.npz")
        astar_path_file = os.path.join(path_dir, f"{base_name}_astar.npz")

        map_data = np.load(map_path)
        x, y, track_width = map_data["x"], map_data["y"], map_data["track_width"]

        q_data = np.load(q_path_file) if os.path.exists(q_path_file) else None
        astar_data = np.load(astar_path_file) if os.path.exists(astar_path_file) else None

        car = Car(x, y, track_width)

        q_path = (q_data["x"], q_data["y"]) if q_data else None
        astar_path = (astar_data["x"], astar_data["y"]) if astar_data else None

        self.render_demo_paths(x, y, car, track_width, q_path=q_path, astar_path=astar_path)

    def render_demo_paths(self, track_x, track_y, car, track_width, q_path=None, astar_path=None):
        show_q = True
        show_astar = True

        simulate_q = False
        simulate_astar = False
        car_q = None
        car_astar = None

        q_max_speed = None
        q_time = None
        astar_max_speed = None
        astar_time = None

        sidebar_width = self.WIDTH - self.map_area_width
        sidebar_x = self.map_area_width

        # Use the button width from main menu
        button_width = self.map_area_width - 450
        button_height = 50
        button_x = sidebar_x + (sidebar_width - button_width) // 2  # Center in sidebar
        button_y_start = 40
        button_spacing = 60
        button_color = (100, 100, 150)  # Muted button color
        button_active_color_q = (80, 130, 180)  # Muted active color for Q
        button_active_color_astar = (205, 100, 50)  # Muted active color for A*

        button_q = pygame.Rect(button_x, button_y_start, button_width, button_height)
        button_astar = pygame.Rect(button_x, button_y_start + button_spacing, button_width, button_height)
        button_sim_q = pygame.Rect(button_x, button_y_start + 2 * button_spacing, button_width, button_height)
        button_sim_astar = pygame.Rect(button_x, button_y_start + 3 * button_spacing, button_width, button_height)
        button_back = pygame.Rect(button_x, self.HEIGHT - button_height - 20, button_width,
                                  button_height)  # back button

        # Prepare track edges
        left_x, left_y,right_x, right_y = self._compute_track_edges(track_x, track_y, track_width)

        # Scaling
        all_x = np.concatenate([left_x, right_x])
        all_y = np.concatenate([left_y, right_y])
        if q_path:
            all_x = np.concatenate([all_x, q_path[0]])
            all_y = np.concatenate([all_y, q_path[1]])
        if astar_path:
            all_x = np.concatenate([all_x, astar_path[0]])
            all_y = np.concatenate([all_y, astar_path[1]])

        transform = self._compute_scaling_and_transform(
            all_x, all_y,
            self.map_area_width * 0.95,
            self.HEIGHT * 0.9,
            self.map_area_width // 2,
            self.HEIGHT // 2
        )

        clock = pygame.time.Clock()
        start_ticks = None

        while True:
            self.screen.fill(MAP_BG)
            pygame.draw.rect(self.screen, SIDEBAR_BG, (sidebar_x, 0, sidebar_width, self.HEIGHT))

            # Buttons
            pygame.draw.rect(self.screen, button_active_color_q if show_q else button_color, button_q)
            pygame.draw.rect(self.screen, button_active_color_astar if show_astar else button_color, button_astar)
            pygame.draw.rect(self.screen, (80, 180, 180), button_sim_q)
            pygame.draw.rect(self.screen, (205, 130, 80), button_sim_astar)
            pygame.draw.rect(self.screen, BACK_BUTTON_COLOR, button_back)

            # Draw centered texts
            self.draw_text_centered("Toggle Q-Learning", button_q)
            self.draw_text_centered("Toggle A*", button_astar)
            self.draw_text_centered("Simulate Q-Learning", button_sim_q)
            self.draw_text_centered("Simulate A*", button_sim_astar)
            self.draw_text_centered("Back to Menu", button_back)

            # Track
            track_poly = list(zip(left_x, left_y)) + list(zip(reversed(right_x), reversed(right_y)))
            pygame.draw.polygon(self.screen, TRACK_GREEN, [transform(px, py) for px, py in track_poly])
            pygame.draw.lines(self.screen, TRACK_LINE, False, [transform(px, py) for px, py in zip(track_x, track_y)],
                              2)

            # Draw paths
            if q_path and (show_q or simulate_q):
                pygame.draw.lines(self.screen, (0, 200, 255), False,
                                  [transform(px, py) for px, py in zip(q_path[0], q_path[1])], 4)
            if astar_path and (show_astar or simulate_astar):
                pygame.draw.lines(self.screen, ASTAR_LINE, False,
                                  [transform(px, py) for px, py in zip(astar_path[0], astar_path[1])], 4)

            # Simulate Q-learning
            if simulate_q and car_q and not car_q.is_finished():
                if start_ticks is None:
                    start_ticks = pygame.time.get_ticks()
                car_q.move_car()
                pos = car_q.get_position()
                if pos:
                    pygame.draw.circle(self.screen, CAR_COLOR, transform(*pos), 6)
            elif simulate_q and car_q and car_q.is_finished() and q_max_speed is None:
                elapsed = (pygame.time.get_ticks() - start_ticks) / 1000
                q_time = elapsed
                q_max_speed = car_q.max_speed_reached * 3.6
                simulate_q = False
                start_ticks = None

            # Simulate A*
            if simulate_astar and car_astar and not car_astar.is_finished():
                if start_ticks is None:
                    start_ticks = pygame.time.get_ticks()
                car_astar.move_car()
                pos = car_astar.get_position()
                if pos:
                    pygame.draw.circle(self.screen, (255, 255, 0), transform(*pos), 6)
            elif simulate_astar and car_astar and car_astar.is_finished() and astar_max_speed is None:
                elapsed = (pygame.time.get_ticks() - start_ticks) / 1000
                astar_time = elapsed
                astar_max_speed = car_astar.max_speed_reached * 3.6
                simulate_astar = False
                start_ticks = None

            # Display metrics
            y_q = 400
            self.screen.blit(self.font.render("Q-Learning Results", True, TEXT_COLOR), (sidebar_x + 40, y_q))
            self.screen.blit(
                self.font.render(f"Max Speed: {q_max_speed:.1f} km/h" if q_max_speed else "Max Speed: --", True,
                                 TEXT_COLOR), (sidebar_x + 40, y_q + 30))
            self.screen.blit(self.font.render(f"Time: {q_time:.2f} s" if q_time else "Time: --", True, TEXT_COLOR),
                             (sidebar_x + 40, y_q + 60))

            y_a = 500
            self.screen.blit(self.font.render("A* Results", True, TEXT_COLOR), (sidebar_x + 40, y_a))
            self.screen.blit(
                self.font.render(f"Max Speed: {astar_max_speed:.1f} km/h" if astar_max_speed else "Max Speed: --", True,
                                 TEXT_COLOR), (sidebar_x + 40, y_a + 30))
            self.screen.blit(
                self.font.render(f"Time: {astar_time:.2f} s" if astar_time else "Time: --", True, TEXT_COLOR),
                (sidebar_x + 40, y_a + 60))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if button_q.collidepoint(event.pos):
                        show_q = not show_q
                    elif button_astar.collidepoint(event.pos):
                        show_astar = not show_astar
                    elif button_sim_q.collidepoint(event.pos):
                        simulate_q = True
                        simulate_astar = False
                        show_q = True
                        show_astar = False
                        car_q = Car(q_path[0], q_path[1], track_width)
                        q_max_speed = None
                        q_time = None
                        start_ticks = None
                    elif button_sim_astar.collidepoint(event.pos):
                        simulate_astar = True
                        simulate_q = False
                        show_astar = True
                        show_q = False
                        car_astar = Car(astar_path[0], astar_path[1], track_width)
                        astar_max_speed = None
                        astar_time = None
                        start_ticks = None
                    elif button_back.collidepoint(event.pos):
                        return
                    elif event.pos[0] < sidebar_x:
                        return

    def draw_text_centered(self, text, rect, color=TEXT_COLOR):
        text_surface = self.font.render(text, True, color)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def show_demo_preview(self):
        demo_dir = "/home/priut/Documents/disertatie/RaceLineOptimization/demo_maps"
        map_files = sorted([
            f for f in os.listdir(demo_dir)
            if f.endswith(".npz") and not f.endswith("_q.npz") and not f.endswith("_astar.npz")
        ])[:4]  # Limit to 4

        if not map_files:
            print("No demo maps found.")
            return

        maps = []
        for filename in map_files:
            data = np.load(os.path.join(demo_dir, filename))
            maps.append((data["x"], data["y"], data["track_width"]))

        selected_file = self.display_map_grid(maps, map_files)
        if selected_file:
            self.run_demo_with_paths("/home/priut/Documents/disertatie/RaceLineOptimization/demo_maps/" + selected_file)
