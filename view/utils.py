import numpy as np

def compute_scaling_and_transform(all_x, all_y, target_width, target_height, center_x, center_y, float = False):
    scale_x = target_width / (np.max(all_x) - np.min(all_x) + 1e-5)
    scale_y = target_height / (np.max(all_y) - np.min(all_y) + 1e-5)
    scale = min(scale_x, scale_y)
    offset_x = center_x - ((np.min(all_x) + np.max(all_x)) / 2) * scale
    offset_y = center_y - ((np.min(all_y) + np.max(all_y)) / 2) * scale

    def transform(px, py):
        if float:
            return px * scale + offset_x, py * scale + offset_y
        else:
            return int(px * scale + offset_x), int(py * scale + offset_y)

    return transform