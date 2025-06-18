import numpy as np
import pygame
import tkinter as tk
from tkinter import filedialog
from model.track import Track
from view.colors import MAP_BG, SIDEBAR_BG, TRACK_GREEN, TRACK_LINE, TEXT_COLOR, BUTTON_COLOR, BACK_BUTTON_COLOR, \
    CONTRAST_TEXT_COLOR


class MapPreviewer:
    """
    Provides a Pygame-based interface for previewing, saving, and regenerating race tracks.
    """
    def __init__(self, screen, font, clock, width=1200, height=700):
        self.screen = screen
        self.font = font
        self.clock = clock
        self.WIDTH = width
        self.HEIGHT = height
        self.map_area_width = 850
        self.sidebar_width = self.WIDTH - self.map_area_width

    def compute_arc_length(self, x, y):
        """Computes the arc length of a path defined by x and y coordinates."""
        points = np.column_stack((x, y))
        diffs = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        return np.sum(segment_lengths)

    def preview_map(self, visualizer, track):
        """
        Displays the map and UI for previewing, regenerating, or saving the track.
        """
        def draw_button(label, y_pos, btncolor=BUTTON_COLOR, txtcolor=TEXT_COLOR):
            btn_rect = pygame.Rect(self.map_area_width + 20, y_pos, self.sidebar_width - 40, 50)
            pygame.draw.rect(self.screen, btncolor, btn_rect)
            text = self.font.render(label, True, txtcolor)
            text_rect = text.get_rect(center=btn_rect.center)
            self.screen.blit(text, text_rect)
            return btn_rect

        track.scale_map_to_left_area(self.map_area_width, self.HEIGHT)
        left_x, left_y, right_x, right_y = track.compute_boundaries()

        while True:
            self.screen.fill(MAP_BG)
            pygame.draw.rect(self.screen, SIDEBAR_BG, (self.map_area_width, 0, self.sidebar_width, self.HEIGHT))

            # Draw track polygon and centerline
            map_poly = list(zip(left_x, left_y)) + list(zip(right_x[::-1], right_y[::-1]))
            pygame.draw.polygon(self.screen, TRACK_GREEN, map_poly)
            pygame.draw.lines(self.screen, TRACK_LINE, False, list(zip(track.x, track.y)), 2)

            # Sidebar title
            title = self.font.render("Track Preview", True, TEXT_COLOR)
            self.screen.blit(title, (self.map_area_width + 30, 40))

            pygame.draw.line(self.screen, (80, 80, 100),
                             (self.map_area_width + 20, 80), (self.WIDTH - 20, 80), 2)

            # Track info
            track_length_m = self.compute_arc_length(track.x, track.y)
            approx_length_km = track_length_m / 1000
            track_info_lines = [
                f"Length: {approx_length_km:.2f} km",
                f"Width: {int(track.width)} m"
            ]
            start_y = 100
            line_spacing = 30
            for i, line in enumerate(track_info_lines):
                info_text = self.font.render(line, True, TEXT_COLOR)
                self.screen.blit(info_text, (self.map_area_width + 40, start_y + i * line_spacing))

            pygame.draw.line(self.screen, (80, 80, 100),
                             (self.map_area_width + 20, start_y + len(track_info_lines) * line_spacing + 10),
                             (self.WIDTH - 20, start_y + len(track_info_lines) * line_spacing + 10), 2)

            # Sidebar buttons
            regenerate_btn = draw_button("Regenerate", 200)
            save_btn = draw_button("Save", 270)
            back_btn = draw_button("Back to Menu", self.HEIGHT - 60, BACK_BUTTON_COLOR, CONTRAST_TEXT_COLOR)

            pygame.display.flip()
            self.clock.tick(30)

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if regenerate_btn.collidepoint(event.pos):
                        track = Track()
                        x, y = track.scale_map_to_left_area(self.map_area_width, self.HEIGHT)
                        left_x, left_y, right_x, right_y = track.compute_boundaries()

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