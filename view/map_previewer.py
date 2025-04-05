import numpy as np
import pygame
import tkinter as tk
from tkinter import filedialog
from model.track import Track
from view.colors import MAP_BG, SIDEBAR_BG, TRACK_GREEN, TRACK_LINE, TEXT_COLOR, BUTTON_COLOR

class MapPreviewer:
    def __init__(self, screen, font, clock, width=1200, height=700):
        self.screen = screen
        self.font = font
        self.clock = clock
        self.WIDTH = width
        self.HEIGHT = height
        self.map_area_width = 850
        self.sidebar_width = self.WIDTH - self.map_area_width

    def scale_map_to_left_area(self, x, y, margin=50):
        drawable_width = self.map_area_width - 2 * margin
        drawable_height = self.HEIGHT - 2 * margin

        x = (x - np.min(x)) / (np.max(x) - np.min(x)) * drawable_width + margin
        y = (y - np.min(y)) / (np.max(y) - np.min(y)) * drawable_height + margin
        x += margin // 2
        return x, y

    def preview_map(self, visualizer, x, y, track_width):
        def draw_button(label, y_pos):
            btn_rect = pygame.Rect(self.map_area_width + 20, y_pos, self.sidebar_width - 40, 50)
            pygame.draw.rect(self.screen, BUTTON_COLOR, btn_rect)
            text = self.font.render(label, True, TEXT_COLOR)
            text_rect = text.get_rect(center=btn_rect.center)
            self.screen.blit(text, text_rect)
            return btn_rect

        def compute_boundaries(x, y, width):
            dx = np.gradient(x)
            dy = np.gradient(y)
            length = np.sqrt(dx ** 2 + dy ** 2)
            length[length == 0] = 1
            dx /= length
            dy /= length
            left_x = x + (width / 2) * dy
            left_y = y - (width / 2) * dx
            right_x = x - (width / 2) * dy
            right_y = y + (width / 2) * dx
            return left_x, left_y, right_x, right_y

        x, y = self.scale_map_to_left_area(x, y)
        left_x, left_y, right_x, right_y = compute_boundaries(x, y, track_width)

        while True:
            self.screen.fill(MAP_BG)
            pygame.draw.rect(self.screen, SIDEBAR_BG, (self.map_area_width, 0, self.sidebar_width, self.HEIGHT))

            map_poly = list(zip(left_x, left_y)) + list(zip(right_x[::-1], right_y[::-1]))
            pygame.draw.polygon(self.screen, TRACK_GREEN, map_poly)
            pygame.draw.lines(self.screen, TRACK_LINE, False, list(zip(x, y)), 2)

            # Sidebar UI
            title = self.font.render("Track Preview", True, TEXT_COLOR)
            self.screen.blit(title, (self.map_area_width + 30, 40))

            info_y = 90
            line_y_spacing = 25
            pygame.draw.line(self.screen, (80, 80, 100), (self.map_area_width + 20, info_y - 10),
                             (self.WIDTH - 20, info_y - 10), 2)

            num_points = len(x)
            approx_length_km = num_points * 0.005
            track_width_m = int(track_width)
            track_info_lines = [
                f"Length: {approx_length_km:.2f} km",
                f"Points: {num_points}",
                f"Width: {track_width_m} m"
            ]
            for i, line in enumerate(track_info_lines):
                info_text = self.font.render(line, True, TEXT_COLOR)
                self.screen.blit(info_text, (self.map_area_width + 40, info_y + i * line_y_spacing))

            pygame.draw.line(self.screen, (80, 80, 100),
                             (self.map_area_width + 20, info_y + 3 * line_y_spacing + 5),
                             (self.WIDTH - 20, info_y + 3 * line_y_spacing + 5), 2)

            regenerate_btn = draw_button("Regenerate", 200)
            save_btn = draw_button("Save", 270)
            back_btn = draw_button("Back to Menu", 340)

            pygame.display.flip()
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if regenerate_btn.collidepoint(event.pos):
                        track = Track()
                        x_new, y_new, track_width = track.generate_map()
                        x, y = self.scale_map_to_left_area(x_new, y_new)
                        left_x, left_y, right_x, right_y = compute_boundaries(x, y, track_width)

                    elif save_btn.collidepoint(event.pos):
                        try:
                            root = tk.Tk()
                            root.withdraw()
                            root.wm_attributes('-topmost', 1)
                            filepath = filedialog.asksaveasfilename(
                                defaultextension=".npz",
                                filetypes=[("NumPy Zip Archive", "*.npz")],
                                title="Save Track As"
                            )
                            root.destroy()
                            if filepath:
                                np.savez(filepath, x=x, y=y, track_width=track_width)
                                print(f"Track saved to {filepath}")
                            else:
                                print("Save cancelled.")
                        except Exception as e:
                            print(f"Error saving track: {e}")

                    elif back_btn.collidepoint(event.pos):
                        return
