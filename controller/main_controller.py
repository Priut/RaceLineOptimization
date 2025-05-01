import numpy as np
import pygame
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Point, Polygon

from model.car import Car
from model.track import Track
from model.racing_env import RacingEnv
from model.agent import QLearningAgent
from model.path_planning.a_star_planner import AStarPlanner
from model.path_planning.grid_builder import GridBuilder
from view.visualization import Visualizer
import scipy.interpolate as interp


def run_qlearning_simulation(visualizer, x_scaled, y_scaled, track_width):
    """
    Trains a Q-learning agent on the current track and visualizes the best path.
    """
    dx = np.gradient(x_scaled)
    dy = np.gradient(y_scaled)
    length = np.sqrt(dx ** 2 + dy ** 2)
    dx /= np.where(length == 0, 1, length)
    dy /= np.where(length == 0, 1, length)

    left_x = x_scaled + (track_width / 2) * dy
    left_y = y_scaled - (track_width / 2) * dx
    right_x = x_scaled - (track_width / 2) * dy
    right_y = y_scaled + (track_width / 2) * dx

    env = RacingEnv(x_scaled, y_scaled, track_width)
    visualizer.current_env = env
    agent = QLearningAgent()
    best_reward = -float('inf')
    best_path = ([], [])

    for episode in range(50):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if total_reward > best_reward:
            best_reward = total_reward
            best_path = (env.path_x.copy(), env.path_y.copy())

    print("Qlearning length = " + str(len(best_path[0])))
    print(len(best_path[0]))
    print(best_path[0][1])

    # ⚡ Clean the path to ensure floats (not numpy arrays)
    clean_best_x = [float(x) for x in best_path[0]]
    clean_best_y = [float(y) for y in best_path[1]]

    print(f"Length x: {len(clean_best_x)}, Length y: {len(clean_best_y)}")
    print(f"Sample x: {clean_best_x[:5]}")
    print(f"Sample y: {clean_best_y[:5]}")
    simple_x, simple_y = simplify_racing_line(clean_best_x, clean_best_y, tolerance=2.0)
    smooth_x, smooth_y = smooth_path(simple_x, simple_y, smoothing=50)
    q_x, q_y = clip_path_to_track(smooth_x, smooth_y, left_x, left_y, right_x, right_y)

    # 🚗 Create car with smooth path
    car = Car(q_x, q_y, track_width)

    # 🏁 Pass the smoothed path for drawing
    visualizer.render_q_agent(x_scaled, y_scaled, car, track_width, best_line=(q_x, q_y))



def smooth_path_q(x, y, smooth_factor=0.01):
    # Use B-spline smoothing
    t = np.linspace(0, 1, len(x))
    spl_x = interp.UnivariateSpline(t, x, s=smooth_factor * len(x))
    spl_y = interp.UnivariateSpline(t, y, s=smooth_factor * len(y))

    t_fine = np.linspace(0, 1, len(x) * 3)  # More points for finer curve
    smooth_x = spl_x(t_fine)
    smooth_y = spl_y(t_fine)
    return smooth_x.tolist(), smooth_y.tolist()


def smooth_path(x, y, smoothing=100.0, num_points=2000):
    """
    Smooths a path using B-spline interpolation.
    """
    x = np.array(x)
    y = np.array(y)

    if len(x) < 4:
        return x, y

    dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    u = np.append([0], np.cumsum(dist))
    u /= u[-1]

    tck, _ = splprep([x, y], u=u, s=smoothing)
    u_fine = np.linspace(0, 1, num_points)
    return splev(u_fine, tck)


def simplify_racing_line(x, y, tolerance=2.0):
    """
    Simplifies the A* path using a geometric tolerance while preserving shape.
    """
    coords = list(zip(x, y))
    line = LineString(coords)
    simplified = line.simplify(tolerance, preserve_topology=False)
    return simplified.xy

def clip_path_to_track(a_star_x, a_star_y, left_x, left_y, right_x, right_y, smooth_factor=0.002):
    """
    Clips path to stay inside track, then applies constrained smoothing.
    """
    from shapely.geometry import LineString, Point, Polygon
    from scipy.interpolate import splprep, splev
    import numpy as np

    # 1. Build track polygon
    track_polygon = Polygon(list(zip(left_x, left_y)) + list(zip(reversed(right_x), reversed(right_y))))

    # 2. First: Clip points
    clipped_x, clipped_y = [], []
    for x, y in zip(a_star_x, a_star_y):
        pt = Point(x, y)
        if track_polygon.contains(pt):
            clipped_x.append(x)
            clipped_y.append(y)
        else:
            nearest = track_polygon.exterior.interpolate(track_polygon.exterior.project(pt))
            clipped_x.append(nearest.x)
            clipped_y.append(nearest.y)

    # 3. Then: Smooth path lightly
    u = np.linspace(0, 1, len(clipped_x))
    tck, _ = splprep([clipped_x, clipped_y], u=u, s=smooth_factor * len(clipped_x))
    u_fine = np.linspace(0, 1, len(clipped_x))
    smooth_x, smooth_y = splev(u_fine, tck)

    # 4. Make sure smoothed points are still inside track
    final_x, final_y = [], []
    for x, y in zip(smooth_x, smooth_y):
        pt = Point(x, y)
        if track_polygon.contains(pt):
            final_x.append(x)
            final_y.append(y)
        else:
            nearest = track_polygon.exterior.interpolate(track_polygon.exterior.project(pt))
            final_x.append(nearest.x)
            final_y.append(nearest.y)

    return final_x, final_y


def run_astar_simulation(visualizer, x_scaled, y_scaled, track_width):
    """
    Runs the A* simulation and visualization for the current track layout.
    """
    grid_builder = GridBuilder(visualizer.map_area_width, visualizer.HEIGHT, resolution=4)

    # Compute track edges
    dx = np.gradient(x_scaled)
    dy = np.gradient(y_scaled)
    length = np.sqrt(dx ** 2 + dy ** 2)
    dx /= np.where(length == 0, 1, length)
    dy /= np.where(length == 0, 1, length)

    left_x = x_scaled + (track_width / 2) * dy
    left_y = y_scaled - (track_width / 2) * dx
    right_x = x_scaled - (track_width / 2) * dy
    right_y = y_scaled + (track_width / 2) * dx

    grid_builder.mark_obstacles_from_track_edges(left_x, left_y, right_x, right_y)

    # Use centerline car for planning only
    car_planning = Car(x_scaled, y_scaled, track_width)
    tck, _ = splprep([x_scaled, y_scaled], s=0)
    astar = AStarPlanner(grid_builder, car_planning, tck)

    start_point = (x_scaled[0], y_scaled[0])
    mid_point = None
    for offset in range(len(x_scaled) // 2):
        i = (len(x_scaled) // 2 + offset) % len(x_scaled)
        gx, gy = x_scaled[i], y_scaled[i]
        gx_i, gy_i = grid_builder.world_to_grid(gx, gy)
        if grid_builder.is_valid_cell(gx_i, gy_i):
            mid_point = (gx, gy)
            break
    if mid_point is None:
        print("❌ No valid midpoint found")
        return

    path1 = astar.plan(start_point, mid_point)
    path2 = astar.plan(mid_point, start_point)

    full_path = []
    if path1:
        full_path += path1
    if path2:
        full_path += path2[1:]

    if full_path and len(full_path) >= 2:
        path_world = [grid_builder.grid_to_world(n.x, n.y) for n in full_path]
        a_star_x, a_star_y = zip(*path_world)
        a_star_x, a_star_y = simplify_racing_line(a_star_x, a_star_y, tolerance=10.0)
        a_star_x, a_star_y = smooth_path(a_star_x, a_star_y, smoothing=50.0)
        a_star_x, a_star_y = clip_path_to_track(a_star_x, a_star_y, left_x, left_y, right_x, right_y)


        a_star_x_scaled, a_star_y_scaled = visualizer.scale_map_to_left_area(np.array(a_star_x), np.array(a_star_y))

        car_sim = Car(a_star_x, a_star_y, track_width)
        visualizer.render_astar(x_scaled, y_scaled, car_sim, track_width, (a_star_x, a_star_y))
    else:
        print("⚠️ A* failed to build full loop path")

def generate_map_preview(visualizer):
    """
    Generates and displays a new map preview.
    """
    track = Track()
    x_spline, y_spline, track_width = track.generate_map()
    x_scaled, y_scaled = visualizer.scale_map_to_left_area(x_spline, y_spline)
    visualizer.preview_map(x_scaled, y_scaled, track_width)

def run_and_save_qlearning_path(x_scaled, y_scaled, track_width, save_path):
    """
    Trains Q-learning and saves the best path to a file.
    """
    dx = np.gradient(x_scaled)
    dy = np.gradient(y_scaled)
    length = np.sqrt(dx ** 2 + dy ** 2)
    dx /= np.where(length == 0, 1, length)
    dy /= np.where(length == 0, 1, length)

    left_x = x_scaled + (track_width / 2) * dy
    left_y = y_scaled - (track_width / 2) * dx
    right_x = x_scaled - (track_width / 2) * dy
    right_y = y_scaled + (track_width / 2) * dx

    env = RacingEnv(x_scaled, y_scaled, track_width)
    agent = QLearningAgent()
    best_reward = -float('inf')
    best_path = ([], [])

    for episode in range(1500):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if total_reward > best_reward:
            best_reward = total_reward
            best_path = (env.path_x.copy(), env.path_y.copy())

    clean_best_x = [float(x) for x in best_path[0]]
    clean_best_y = [float(y) for y in best_path[1]]

    simple_x, simple_y = simplify_racing_line(clean_best_x, clean_best_y, tolerance=2.0)
    smooth_x, smooth_y = smooth_path(simple_x, simple_y, smoothing=50)
    q_x, q_y = clip_path_to_track(smooth_x, smooth_y, left_x, left_y, right_x, right_y)

    np.savez(save_path, x=q_x, y=q_y)
    print(f"✅ Q-learning path saved to {save_path}")


def run_and_save_astar_path(x_scaled, y_scaled, track_width, save_path, visualizer=None):
    """
    Runs A* planning and saves the smoothed path to a file.
    """
    grid_builder = GridBuilder(visualizer.map_area_width if visualizer else 850, 700, resolution=4)

    dx = np.gradient(x_scaled)
    dy = np.gradient(y_scaled)
    length = np.sqrt(dx ** 2 + dy ** 2)
    dx /= np.where(length == 0, 1, length)
    dy /= np.where(length == 0, 1, length)

    left_x = x_scaled + (track_width / 2) * dy
    left_y = y_scaled - (track_width / 2) * dx
    right_x = x_scaled - (track_width / 2) * dy
    right_y = y_scaled + (track_width / 2) * dx

    grid_builder.mark_obstacles_from_track_edges(left_x, left_y, right_x, right_y)

    car = Car(x_scaled, y_scaled, track_width)
    tck, _ = splprep([x_scaled, y_scaled], s=0)
    astar = AStarPlanner(grid_builder, car, tck)

    start_point = (x_scaled[0], y_scaled[0])
    mid_point = None
    for offset in range(len(x_scaled) // 2):
        i = (len(x_scaled) // 2 + offset) % len(x_scaled)
        gx, gy = x_scaled[i], y_scaled[i]
        gx_i, gy_i = grid_builder.world_to_grid(gx, gy)
        if grid_builder.is_valid_cell(gx_i, gy_i):
            mid_point = (gx, gy)
            break
    if mid_point is None:
        print("❌ No valid midpoint found for A*")
        return

    path1 = astar.plan(start_point, mid_point)
    path2 = astar.plan(mid_point, start_point)

    full_path = []
    if path1:
        full_path += path1
    if path2:
        full_path += path2[1:]

    if full_path and len(full_path) >= 2:
        path_world = [grid_builder.grid_to_world(n.x, n.y) for n in full_path]
        a_star_x, a_star_y = zip(*path_world)
        a_star_x, a_star_y = simplify_racing_line(a_star_x, a_star_y, tolerance=10.0)
        a_star_x, a_star_y = smooth_path(a_star_x, a_star_y, smoothing=50.0)
        a_star_x, a_star_y = clip_path_to_track(a_star_x, a_star_y, left_x, left_y, right_x, right_y)

        np.savez(save_path, x=a_star_x, y=a_star_y)
        print(f"✅ A* path saved to {save_path}")
    else:
        print("⚠️ A* path was not valid and was not saved.")


import os
import numpy as np

def process_all_demo_maps(demo_dir, visualizer):
    """
    Processes all .npz maps in a directory by training Q-learning and A* paths,
    then saves those paths as <mapname>_q.npz and <mapname>_astar.npz.
    """
    demo_files = [f for f in os.listdir(demo_dir) if f.endswith(".npz") and not f.endswith(("_q.npz", "_astar.npz"))]

    if not demo_files:
        print("❌ No demo maps found in the folder.")
        return

    for filename in demo_files:
        full_path = os.path.join(demo_dir, filename)
        base_name = os.path.splitext(filename)[0]

        try:
            data = np.load(full_path)
            x, y, width = data["x"], data["y"], data["track_width"]
        except Exception as e:
            print(f"⚠️ Could not load map {filename}: {e}")
            continue

        print(f"\n▶️ Processing map: {filename}")

        q_path = os.path.join(demo_dir, "paths", f"{base_name}_q.npz")
        run_and_save_qlearning_path(x, y, width, q_path)

        astar_path = os.path.join(demo_dir, "paths", f"{base_name}_astar.npz")
        run_and_save_astar_path(x, y, width, astar_path, visualizer)

    print("\n✅ All maps processed and saved.")

def main():
    """
    Application entry point for running the simulator.
    """
    pygame.init()
    visualizer = Visualizer()
    running = True
    #process_all_demo_maps("/home/priut/Documents/disertatie/RaceLineOptimization/demo_maps", visualizer)

    while running:
        choice = visualizer.show_main_menu()
        if choice == "generate_map":
            generate_map_preview(visualizer)
        elif choice == "simulate":
            while True:
                sim_choice = visualizer.show_simulation_menu()
                track = Track()
                x_spline, y_spline, track_width = track.generate_map()
                x_scaled, y_scaled = visualizer.scale_map_to_left_area(x_spline, y_spline)

                if sim_choice == "qlearning_agent":
                    visualizer.show_loading_screen(x_scaled, y_scaled, track_width, "Training AI... Please wait")
                    run_qlearning_simulation(visualizer, x_scaled, y_scaled, track_width)
                elif sim_choice == "a*":
                    visualizer.show_loading_screen(x_scaled, y_scaled, track_width, "Finding A* Path... Please wait")
                    run_astar_simulation(visualizer, x_scaled, y_scaled, track_width)
                elif sim_choice == "back_to_main_menu":
                    break
        elif choice == "demo":
            visualizer.show_demo_preview()

    pygame.quit()
