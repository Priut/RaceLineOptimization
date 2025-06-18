import os
import numpy as np
import pygame

from model.car import Car
from model.track import Track
from view.map_previewer import MapPreviewer
from view.colors import TRACK_GREEN, TRACK_LINE, CAR_COLOR, SIDEBAR_BG, TEXT_COLOR, MAP_BG, ASTAR_LINE, \
    BACK_BUTTON_COLOR, CONTRAST_TEXT_COLOR
import view.utils as utils


class DemoViewer:
    """
    Handles the visual presentation and simulation of demo tracks in a grid interface.
    Allows users to preview predefined racing maps and run Q-learning and A* path-following simulations,
    displaying performance metrics in real time.
    """
    def __init__(self, screen, font, width, height, map_area_width):
        """Initializes the viewer with the Pygame screen, font, layout dimensions, and sidebar width."""
        self.screen = screen
        self.font = font
        self.WIDTH = width
        self.HEIGHT = height
        self.map_area_width = map_area_width
        self.bold_font = pygame.font.SysFont('Consolas', 18, bold=True)


    def display_map_grid(self, map_data_list, filenames):
        """
        Displays up to four demo maps in a 2x2 grid.
        Each map is clickable; selecting one returns the corresponding filename.
        Includes a "Back" button to return to the previous screen.
        """
        self.screen.fill(MAP_BG)
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

            track = Track(custom_x=x, custom_y=y, custom_width=width)
            left_x, left_y, right_x, right_y = track.compute_boundaries()

            all_x = np.concatenate([left_x, right_x])
            all_y = np.concatenate([left_y, right_y])
            transform = utils.compute_scaling_and_transform(
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
        """
        Loads a demo map and its associated Q-learning and A* paths (if available), then launches the visual demo simulation view.
        """
        base_directory = "/home/priut/Documents/disertatie/RaceLineOptimization/"
        map_path = map_filename  # already full path now
        base_name = os.path.splitext(os.path.basename(map_filename))[0]
        path_dir = os.path.join(base_directory, "demo_maps", "paths")
        q_path_file = os.path.join(path_dir, f"{base_name}_q.npz")
        astar_path_file = os.path.join(path_dir, f"{base_name}_astar.npz")

        map_data = np.load(map_path)
        x, y, track_width = map_data["x"], map_data["y"], map_data["track_width"]
        track = Track(custom_x=x, custom_y=y, custom_width=track_width)

        q_data = np.load(q_path_file) if os.path.exists(q_path_file) else None
        astar_data = np.load(astar_path_file) if os.path.exists(astar_path_file) else None

        car = Car(x, y, track_width)

        q_path = (q_data["x"], q_data["y"]) if q_data else None
        astar_path = (astar_data["x"], astar_data["y"]) if astar_data else None

        self.render_demo_paths(track, car, q_path=q_path, astar_path=astar_path)

    def render_demo_paths(self, track, car, q_path=None, astar_path=None):
        """
        Main interactive view for demo simulations.
        Displays the track and overlays the Q-learning and A* paths.
        Includes buttons to toggle visibility and start path-following simulations.
        Displays real-time and final statistics (max speed, average speed, time, and path length).
        """
        show_q = True
        show_astar = True

        simulate_q = False
        simulate_astar = False
        car_q = None
        car_astar = None

        q_max_speed = None
        q_time = None
        q_path_length = None
        q_avg_speed = None
        astar_max_speed = None
        astar_time = None
        astar_path_length = None
        astar_avg_speed = None

        sidebar_width = self.WIDTH - self.map_area_width
        sidebar_x = self.map_area_width

        # Buttons
        button_width = self.map_area_width - 450
        button_height = 50
        button_x = sidebar_x + (sidebar_width - button_width) // 2
        button_y_start = 40
        button_spacing = 60
        button_color = (100, 100, 150)
        button_active_color_q = (80, 130, 180)
        button_active_color_astar = (205, 100, 50)

        button_q = pygame.Rect(button_x, button_y_start, button_width, button_height)
        button_astar = pygame.Rect(button_x, button_y_start + button_spacing, button_width, button_height)
        button_sim_q = pygame.Rect(button_x, button_y_start + 2 * button_spacing, button_width, button_height)
        button_sim_astar = pygame.Rect(button_x, button_y_start + 3 * button_spacing, button_width, button_height)
        button_back = pygame.Rect(button_x, self.HEIGHT - button_height - 20, button_width,
                                  button_height)

        # Track edges
        left_x, left_y, right_x, right_y = track.compute_boundaries()

        # Scaling
        all_x = np.concatenate([left_x, right_x])
        all_y = np.concatenate([left_y, right_y])
        if q_path:
            all_x = np.concatenate([all_x, q_path[0]])
            all_y = np.concatenate([all_y, q_path[1]])
        if astar_path:
            all_x = np.concatenate([all_x, astar_path[0]])
            all_y = np.concatenate([all_y, astar_path[1]])

        transform = utils.compute_scaling_and_transform(
            all_x, all_y,
            self.map_area_width * 0.95,
            self.HEIGHT * 0.9,
            self.map_area_width // 2,
            self.HEIGHT // 2
        )

        clock = pygame.time.Clock()
        start_ticks = None

        while True:
            clock.tick(60)
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
            self.draw_text_centered("Back to Menu", button_back, color=CONTRAST_TEXT_COLOR)

            # Track
            track_poly = list(zip(left_x, left_y)) + list(zip(reversed(right_x), reversed(right_y)))
            pygame.draw.polygon(self.screen, TRACK_GREEN, [transform(px, py) for px, py in track_poly])
            pygame.draw.lines(self.screen, TRACK_LINE, False, [transform(px, py) for px, py in zip(track.x, track.y)],
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
                q_path_length = sum(np.hypot(np.diff(q_path[0]), np.diff(q_path[1]))) / 1000  # km
                q_avg_speed = (q_path_length / elapsed) * 3600 if elapsed > 0 else 0  # km/h
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
                astar_path_length = sum(np.hypot(np.diff(astar_path[0]), np.diff(astar_path[1]))) / 1000  # km
                astar_avg_speed = (astar_path_length / elapsed) * 3600 if elapsed > 0 else 0  # km/h
                simulate_astar = False
                start_ticks = None

            # Display metrics
            y_q = 300
            pygame.draw.line(self.screen, (80, 80, 100), (sidebar_x + 20, y_q - 10), (self.WIDTH - 20, y_q - 10), 2)
            self.screen.blit(self.bold_font.render("Q-Learning Results", True, TEXT_COLOR), (sidebar_x + 40, y_q))
            pygame.draw.line(self.screen, (80, 80, 100), (sidebar_x + 20, y_q + 28), (self.WIDTH - 20, y_q + 28), 2)
            self.screen.blit(
                self.font.render(f"Max Speed: {q_max_speed:.1f} km/h" if q_max_speed else "Max Speed: --", True,
                                 TEXT_COLOR), (sidebar_x + 40, y_q + 38))
            self.screen.blit(self.font.render(f"Time: {q_time:.2f} s" if q_time else "Time: --", True, TEXT_COLOR),
                             (sidebar_x + 40, y_q + 68))
            self.screen.blit(
                self.font.render(f"Avg Speed: {q_avg_speed:.1f} km/h" if q_avg_speed else "Avg Speed: --", True,
                                 TEXT_COLOR), (sidebar_x + 40, y_q + 98))
            self.screen.blit(
                self.font.render(f"Path Length: {q_path_length:.2f} km" if q_path_length else "Path Length: --", True,
                                 TEXT_COLOR), (sidebar_x + 40, y_q + 128))


            y_a = 470
            pygame.draw.line(self.screen, (80, 80, 100), (sidebar_x + 20, y_a - 10), (self.WIDTH - 20, y_a - 10), 2)
            self.screen.blit(self.bold_font.render("A* Results", True, TEXT_COLOR), (sidebar_x + 40, y_a))
            pygame.draw.line(self.screen, (80, 80, 100), (sidebar_x + 20, y_a + 28), (self.WIDTH - 20, y_a + 28), 2)
            self.screen.blit(
                self.font.render(f"Max Speed: {astar_max_speed:.1f} km/h" if astar_max_speed else "Max Speed: --", True,
                                 TEXT_COLOR), (sidebar_x + 40, y_a + 38))
            self.screen.blit(
                self.font.render(f"Time: {astar_time:.2f} s" if astar_time else "Time: --", True, TEXT_COLOR),
                (sidebar_x + 40, y_a + 68))
            self.screen.blit(
                self.font.render(f"Avg Speed: {astar_avg_speed:.1f} km/h" if astar_avg_speed else "Avg Speed: --", True,
                                 TEXT_COLOR), (sidebar_x + 40, y_a + 98))
            self.screen.blit(
                self.font.render(f"Path Length: {astar_path_length:.2f} km" if astar_path_length else "Path Length: --",
                                 True, TEXT_COLOR), (sidebar_x + 40, y_a + 128))

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
                        car_q = Car(q_path[0], q_path[1], track.width)
                        q_max_speed = None
                        q_time = None
                        q_path_length = None
                        q_avg_speed = None
                        start_ticks = None
                    elif button_sim_astar.collidepoint(event.pos):
                        simulate_astar = True
                        simulate_q = False
                        show_astar = True
                        show_q = False
                        car_astar = Car(astar_path[0], astar_path[1], track.width)
                        astar_max_speed = None
                        astar_time = None
                        astar_path_length = None
                        astar_avg_speed = None
                        start_ticks = None
                    elif button_back.collidepoint(event.pos):
                        return
                    elif event.pos[0] < sidebar_x:
                        return

    def draw_text_centered(self, text, rect, color=TEXT_COLOR):
        """Draws text centered within a given pygame.Rect."""
        text_surface = self.font.render(text, True, color)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    def show_demo_preview(self):
        """
        Searches the demo_maps directory for valid .npz demo files (excluding Q/A* files),
        loads the first four, and displays them in a grid.
        Upon selection, the corresponding demo simulation is run.
        """
        demo_dir = "/home/priut/Documents/disertatie/RaceLineOptimization/demo_maps"
        map_files = sorted([
            f for f in os.listdir(demo_dir)
            if f.endswith(".npz") and not f.endswith("_q.npz") and not f.endswith("_astar.npz")
        ])[:4]

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