import pygame
from model.car import Car
from model.track import Track
from model.racing_env import RacingEnv
from model.agent import QLearningAgent
from view.visualization import Visualizer

def run_simulation(visualizer):
    track = Track()
    x_spline, y_spline, track_width = track.generate_map()
    visualizer.show_training_screen()
    x_spline_scaled, y_spline_scaled = visualizer.scale_points(x_spline, y_spline)

    env = RacingEnv(x_spline_scaled, y_spline_scaled, track_width)
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

        print(f"Episode {episode+1} - Total Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")
        if total_reward > best_reward:
            best_reward = total_reward
            best_path = (env.path_x.copy(), env.path_y.copy())

    car = Car(best_path[0], best_path[1], track_width)
    visualizer.render(x_spline_scaled, y_spline_scaled, car, track_width, best_line=best_path)


def generate_map_preview(visualizer):
    track = Track()
    x_spline, y_spline, track_width = track.generate_map()
    x_scaled, y_scaled = visualizer.scale_points(x_spline, y_spline)
    visualizer.preview_map(x_scaled, y_scaled, track_width)


def main():
    pygame.init()
    visualizer = Visualizer()
    running = True

    while running:
        choice = visualizer.show_main_menu()  # This blocks until a button is clicked

        if choice == "generate_map":
            generate_map_preview(visualizer)

        elif choice == "simulate":
            run_simulation(visualizer)
            running = False  # optional: close after simulation

    pygame.quit()
