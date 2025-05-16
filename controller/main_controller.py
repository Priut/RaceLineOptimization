import numpy as np
import pygame
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Point, Polygon

from view.utils import compute_scaling_and_transform

from model.car import Car
from model.track import Track
from model.racing_env import RacingEnv
from model.agent import QLearningAgent
from model.path_planning.a_star_planner import AStarPlanner
from model.path_planning.grid_builder import GridBuilder
from view.visualization import Visualizer
import scipy.interpolate as interp
import random
import os


def train_qlearning_on_multiple_maps(track_list, episodes=50):
    """
    Trains a Q-learning agent on multiple tracks by randomly alternating between them each episode.

    Args:
        track_list (List[Track]): List of Track objects.
        episodes (int): Number of training episodes.

    Returns:
        QLearningAgent: Trained agent.
    """
    agent = QLearningAgent()

    for episode in range(episodes):
        # Choose a random track each episode
        track = random.choice(track_list)
        env = RacingEnv(track.x, track.y, track.width)

        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state

    return agent

def run_qlearning_simulation(visualizer, track, agent):
    """
    Runs the trained agent on a given track and visualizes the path.
    """
    # Transform track centerline
    # Combine all x/y for consistent scaling
    track_x, track_y = track.x, track.y
    left_x, left_y, right_x, right_y = track.compute_boundaries()
    all_x = list(track_x) + list(left_x) + list(right_x)
    all_y = list(track_y) + list(left_y) + list(right_y)

    center_x = visualizer.simulationViewer.map_area_width // 2
    center_y = visualizer.simulationViewer.HEIGHT // 2

    transform = compute_scaling_and_transform(
        all_x, all_y,
        visualizer.simulationViewer.map_area_width,
        visualizer.simulationViewer.HEIGHT,
        center_x, center_y,
        float=True
    )

    # Transform everything with consistent scale
    track.x, track.y = zip(*[transform(x, y) for x, y in zip(track_x, track_y)])
    left_x, left_y = zip(*[transform(x, y) for x, y in zip(left_x, left_y)])
    right_x, right_y = zip(*[transform(x, y) for x, y in zip(right_x, right_y)])

    env = RacingEnv(track.x, track.y, track.width)
    visualizer.simulationViewer.current_env = env

    state = env.reset()
    done = False

    while not done:
        action = agent.choose_action(state)
        state, _, done = env.step(action)

    best_x, best_y = env.path_x, env.path_y
    simple_x, simple_y = simplify_racing_line(best_x, best_y, tolerance=2.0)
    smooth_x, smooth_y = smooth_path(simple_x, simple_y, smoothing=50)
    q_x, q_y = clip_path_to_track(smooth_x, smooth_y, left_x, left_y, right_x, right_y)

    # Combine all x/y for consistent scaling
    all_x = list(q_x) + list(left_x) + list(right_x)
    all_y = list(q_y) + list(left_y) + list(right_y)

    center_x = visualizer.simulationViewer.map_area_width // 2
    center_y = visualizer.simulationViewer.HEIGHT // 2

    transform = compute_scaling_and_transform(
        all_x, all_y,
        visualizer.simulationViewer.map_area_width,
        visualizer.simulationViewer.HEIGHT,
        center_x, center_y,
        float = True
    )

    # Apply transform to all
    scaled_q_x, scaled_q_y = zip(*[transform(x, y) for x, y in zip(q_x, q_y)])

    car = Car(scaled_q_x, scaled_q_y, track.width)
    visualizer.render_q_agent(track, car, best_line=(scaled_q_x, scaled_q_y))


def run_qlearning_simulation1(visualizer, track):
    """
    Trains a Q-learning agent on the current track and visualizes the best path.
    """
    left_x, left_y, right_x, right_y = track.compute_boundaries()

    env = RacingEnv(track.x, track.y, track.width)
    visualizer.simulationViewer.current_env = env
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

    clean_best_x = [float(x) for x in best_path[0]]
    clean_best_y = [float(y) for y in best_path[1]]
    simple_x, simple_y = simplify_racing_line(clean_best_x, clean_best_y, tolerance=2.0)
    smooth_x, smooth_y = smooth_path(simple_x, simple_y, smoothing=50)
    q_x, q_y = clip_path_to_track(smooth_x, smooth_y, left_x, left_y, right_x, right_y)

    car = Car(q_x, q_y, track.width)
    visualizer.render_q_agent(track, car, best_line=(q_x, q_y))

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


def run_astar_simulation(visualizer, track):
    """
    Runs the A* simulation and visualization for the current track layout.
    """
    grid_builder = GridBuilder(visualizer.map_area_width, visualizer.HEIGHT, resolution=4)

    # Compute track edges
    left_x, left_y, right_x, right_y = track.compute_boundaries()

    grid_builder.mark_obstacles_from_track_edges(left_x, left_y, right_x, right_y)

    # Use centerline car for planning only
    car_planning = Car(track.x, track.y, track.width)
    tck, _ = splprep([track.x, track.y], s=0)
    astar = AStarPlanner(grid_builder, car_planning, tck)

    start_point = (track.x[0], track.y[0])
    mid_point = None
    for offset in range(len(track.x) // 2):
        i = (len(track.x) // 2 + offset) % len(track.x)
        gx, gy = track.x[i], track.y[i]
        gx_i, gy_i = grid_builder.world_to_grid(gx, gy)
        if grid_builder.is_valid_cell(gx_i, gy_i):
            mid_point = (gx, gy)
            break
    if mid_point is None:
        print("No valid midpoint found")
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

        car_sim = Car(a_star_x, a_star_y, track.width)
        visualizer.render_astar(track, car_sim, (a_star_x, a_star_y))
    else:
        print("A* failed to build full loop path")

def run_and_save_qlearning_path(track, save_path, trained_agent):
    """
    Uses a trained Q-learning agent to generate the best path on a given track and saves it to a file.

    Args:
        track (Track): The track on which to evaluate the agent.
        save_path (str): Path to save the computed best path.
        trained_agent (QLearningAgent): The already trained agent to use.
    """
    left_x, left_y, right_x, right_y = track.compute_boundaries()

    env = RacingEnv(track.x, track.y, track.width)
    state = env.reset()
    done = False

    while not done:
        action = trained_agent.choose_action(state)
        state, _, done = env.step(action)

    best_x, best_y = env.path_x, env.path_y

    # Post-process the path
    simple_x, simple_y = simplify_racing_line(best_x, best_y, tolerance=2.0)
    smooth_x, smooth_y = smooth_path(simple_x, simple_y, smoothing=50)
    q_x, q_y = clip_path_to_track(smooth_x, smooth_y, left_x, left_y, right_x, right_y)

    np.savez(save_path, x=q_x, y=q_y)
    print(f"Q-learning path saved to {save_path}")

def run_and_save_astar_path(track, save_path):
    """
    Runs A* planning and saves the smoothed path to a file.
    """
    grid_builder = GridBuilder(850, 700, resolution=4)

    left_x, left_y, right_x, right_y = track.compute_boundaries()
    grid_builder.mark_obstacles_from_track_edges(left_x, left_y, right_x, right_y)

    car = Car(track.x, track.y, track.width)
    tck, _ = splprep([track.x, track.y], s=0)
    astar = AStarPlanner(grid_builder, car, tck)

    start_point = (track.x[0], track.y[0])
    mid_point = None
    for offset in range(len(track.x) // 2):
        i = (len(track.x) // 2 + offset) % len(track.x)
        gx, gy = track.x[i], track.y[i]
        gx_i, gy_i = grid_builder.world_to_grid(gx, gy)
        if grid_builder.is_valid_cell(gx_i, gy_i):
            mid_point = (gx, gy)
            break
    if mid_point is None:
        print("No valid midpoint found for A*")
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
        print(f"A* path saved to {save_path}")
    else:
        print("A* path was not valid and was not saved.")

def process_all_demo_maps(demo_dir, visualizer, trained_agent):
    demo_files = [f for f in os.listdir(demo_dir) if f.endswith(".npz") and not f.endswith(("_q.npz", "_astar.npz"))]

    if not demo_files:
        print("No demo maps found in the folder.")
        return

    for filename in demo_files:
        full_path = os.path.join(demo_dir, filename)
        base_name = os.path.splitext(filename)[0]

        try:
            data = np.load(full_path)
            x, y, width = data["x"], data["y"], data["track_width"]
            track = Track(custom_x=x, custom_y=y, custom_width=width)
        except Exception as e:
            print(f"Could not load map {filename}: {e}")
            continue

        print(f"\nProcessing map: {filename}")

        q_path = os.path.join(demo_dir, "paths", f"{base_name}_q.npz")
        run_and_save_qlearning_path(track, q_path, trained_agent)

        astar_path = os.path.join(demo_dir, "paths", f"{base_name}_astar.npz")
        run_and_save_astar_path(track, astar_path)

    print("\nAll maps processed and saved.")



def main():
    pygame.init()
    visualizer = Visualizer()
    running = True

    agent_path = "trained_agent.pkl"
    trained_agent = None

    # Load trained agent if it exists
    if os.path.exists(agent_path):
        trained_agent = QLearningAgent.load(agent_path)
        print("[INFO] Loaded trained agent from disk.")
    else:
        print("[INFO] Training new agent...")
        training_tracks = [Track() for _ in range(20)]
        trained_agent = train_qlearning_on_multiple_maps(training_tracks, episodes=2000)
        trained_agent.save(agent_path)
        print(f"[INFO] Agent trained and saved to {agent_path}.")

    demo_dir = "/home/priut/Documents/disertatie/RaceLineOptimization/demo_maps"
    #process_all_demo_maps(demo_dir, visualizer, trained_agent)

    while running:
        choice = visualizer.show_main_menu()

        if choice == "generate_map":
            track = Track()
            visualizer.preview_map(track)

        elif choice == "simulate":
            while True:
                sim_choice = visualizer.show_simulation_menu()

                if sim_choice == "qlearning_agent":
                    eval_track = Track()
                    run_qlearning_simulation(visualizer, eval_track, trained_agent)

                elif sim_choice == "a*":
                    track = Track()
                    visualizer.show_loading_screen(track, "Finding A* Path... Please wait")
                    run_astar_simulation(visualizer, track)

                elif sim_choice == "back_to_main_menu":
                    break

        elif choice == "demo":
            visualizer.show_demo_preview()

    pygame.quit()

