def convert_path_to_world(path, grid_builder):
    """
    Converts a list of Node objects (grid cells) to world coordinates (float x, y)
    """
    world_coords = []
    for node in path:
        wx, wy = grid_builder.grid_to_world(node.x, node.y)
        world_coords.append((wx, wy))
    return world_coords
