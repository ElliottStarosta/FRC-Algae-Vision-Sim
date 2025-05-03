import pygame
import sys
import math
import numpy as np
from enum import Enum

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 600
ROBOT_COLOR = (30, 144, 255)
ALGAE_COLOR = (50, 205, 50)
INTERCEPT_COLOR = (255, 0, 0)
PREDICTION_COLOR = (255, 255, 0)
BG_COLOR = (20, 20, 20)
TEXT_COLOR = (255, 255, 255)
LINE_COLOR = (100, 100, 100, 128)

COLLISION_THRESHOLD = 30

STOP_TIME_THRESHOLD = 10.0  # seconds


class SimulationCase(Enum):
    NORMAL = "Normal Case"
    ALGAE_RESTS = "Algae comes to rest before interception"
    NO_MAX_VELOCITY = "Robot doesn't reach max velocity"
    ALGAE_IN_FRONT = "Algae moving in front of robot"
    FAST_ALGAE = "Algae traveling faster than robot's max velocity"
    NEGATIVE_INITIAL = "Robot's initial velocity is negative"
    PLAYGROUND = "Playground Mode (Custom Setup)"


class Simulator:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Robot-Algae Interception Simulator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.large_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.paused = False  # Initialize paused attribute here

        # Initialize all editing flags to prevent reference errors
        self.editing_robot = False
        self.editing_algae = False
        self.editing_robot_vel = False
        self.editing_algae_vel = False

        # Initialize default values first
        self.robot_pos = np.array([200.0, HEIGHT / 2])
        self.algae_pos = np.array([800.0, HEIGHT / 2])
        self.robot_vel = np.array([0.0, 0.0])
        self.algae_vel = np.array([-40.0, 0.0])
        self.robot_acc = 40.0
        self.robot_max_vel = 400.0
        self.algae_decel = 0.0
        self.time = 0.0
        self.dt = 0.016  # ~60 FPS
        self.path_points_robot = [self.robot_pos.copy()]
        self.path_points_algae = [self.algae_pos.copy()]
        self.intercept_point = None
        self.intercept_time = None
        self.intercept_calculated = False

        # Additional playground mode variables
        self.playground_mode = False
        self.placing_robot = False
        self.placing_algae = False
        self.setting_robot_vel = False
        self.setting_algae_vel = False
        self.setup_completed = False
        self.vel_start_pos = None
        self.setup_stage = 0  # Track the setup stage in playground mode

        self.reset_simulation(SimulationCase.NORMAL)

    """
    Reset the simulation to a predetermined configuration based on the selected case.
    Args:
        case (SimulationCase): The simulation scenario to load with predefined parameters.
    """
    def reset_simulation(self, case):
        #  Way to save the previous state
        if hasattr(self, "robot_pos"):
            # Save previous state before resetting
            self.prev_robot_pos = self.robot_pos.copy()
            self.prev_algae_pos = self.algae_pos.copy()
            self.prev_robot_vel = self.robot_vel.copy()
            self.prev_algae_vel = self.algae_vel.copy()
            self.prev_robot_acc = self.robot_acc
            self.prev_robot_max_vel = self.robot_max_vel
            self.prev_algae_decel = self.algae_decel

        # Reset editing flags
        self.editing_robot = False
        self.editing_algae = False
        self.editing_robot_vel = False
        self.editing_algae_vel = False

        # Reset to default values first
        self.robot_pos = np.array([200.0, HEIGHT / 2])
        self.algae_pos = np.array([800.0, HEIGHT / 2])
        self.robot_vel = np.array([0.0, 0.0])
        self.algae_vel = np.array([-100.0, 0.0])
        self.robot_acc = 100.0
        self.robot_max_vel = 200.0
        self.algae_decel = 0.0
        self.fixed_target = None  # Reset the fixed target

        # Additional playground mode variables
        self.playground_mode = False
        self.placing_robot = False
        self.placing_algae = False
        self.setting_robot_vel = False
        self.setting_algae_vel = False
        self.setup_completed = False
        self.vel_start_pos = None
        self.setup_stage = 0  # Track the setup stage in playground mode

        # Adjust parameters based on case
        if case == SimulationCase.ALGAE_RESTS:
            # Algae stops after some distance - now at different y-coordinate
            self.algae_pos = np.array([800.0, HEIGHT / 2 - 120])  # Above center line
            self.robot_pos = np.array([200.0, HEIGHT / 2 + 80])  # Below center line
            self.algae_vel = np.array([-100.0, 20.0])  # Angled movement
            self.algae_decel = 50.0  # Algae decelerates

        if case == SimulationCase.NO_MAX_VELOCITY:
            # Set parameters where robot won't reach max velocity
            self.robot_pos = np.array([340.0, HEIGHT / 2 - 20])
            self.algae_pos = np.array([620.0, HEIGHT / 2])
            self.robot_max_vel = 300.0

        if case == SimulationCase.ALGAE_IN_FRONT:
            # Algae starts in front of robot - different y-coordinate
            self.robot_pos = np.array([500.0, HEIGHT / 2 + 60])  # Below center
            self.algae_pos = np.array([300.0, HEIGHT / 2 - 100])  # Above center
            self.algae_vel = np.array([30.0, 20.0])  # Moving diagonally

        if case == SimulationCase.FAST_ALGAE:
            # Algae moves faster than robot's max speed
            self.robot_pos = np.array([200.0, HEIGHT / 2 - 90])  # Above center
            self.algae_pos = np.array([800.0, HEIGHT / 2 + 150])  # Below center
            self.algae_vel = np.array([-400.0, -50.0])  # Diagonal fast movement

        if case == SimulationCase.NEGATIVE_INITIAL:
            # Robot initially moving away from algae
            self.robot_vel = np.array([-50.0, 0.0])

        if case == SimulationCase.PLAYGROUND:
            # Enter playground setup mode
            self.playground_mode = True
            self.setup_stage = 1  # Start at stage 1: placing robot
            self.placing_robot = True
            self.paused = True  # Start paused in setup mode
            
            # NEW: Initialize storage for initial positions (for W key reset)
            if hasattr(self, "initial_setup_robot_pos"):
                delattr(self, "initial_setup_robot_pos")
            if hasattr(self, "initial_setup_algae_pos"):
                delattr(self, "initial_setup_algae_pos")

        # Common settings for all cases
        self.current_case = case
        self.time = 0.0
        self.dt = 0.016  # ~60 FPS
        self.running = True
        self.intercept_calculated = False
        self.intercept_point = None
        self.intercept_time = None
        self.path_points_robot = [self.robot_pos.copy()]
        self.path_points_algae = [self.algae_pos.copy()]

        # Only calculate intercept if not in playground mode or setup is completed
        if not self.playground_mode or self.setup_completed:
            self.calculate_intercept()
        
        

    """
    Restore the simulation to its previous state before the last reset.
    This allows users to undo changes and return to a known configuration.

    Returns:
        bool: True if previous state was successfully restored, False otherwise.
    """
    def restore_previous_state(self):
        """Restore the previous simulation state"""
        if hasattr(self, "prev_robot_pos"):
            self.robot_pos = self.prev_robot_pos.copy()
            self.algae_pos = self.prev_algae_pos.copy()
            self.robot_vel = self.prev_robot_vel.copy()
            self.algae_vel = self.prev_algae_vel.copy()
            self.robot_acc = self.prev_robot_acc
            self.robot_max_vel = self.prev_robot_max_vel
            self.algae_decel = self.prev_algae_decel

            # Reset paths and time
            self.time = 0.0
            self.path_points_robot = [self.robot_pos.copy()]
            self.path_points_algae = [self.algae_pos.copy()]

            # Recalculate the interception
            self.calculate_intercept()
            return True
        return False

    """
    Calculate the expected interception point and time using proper 2D motion equations
    considering robot acceleration, max velocity, and algae deceleration.
    Sets the intercept_point, intercept_time, and intercept_calculated attributes.
    """
    def calculate_intercept(self):
        # Starting positions and velocities
        robot_pos = self.robot_pos.copy()
        robot_vel = self.robot_vel.copy()
        algae_pos = self.algae_pos.copy()
        algae_vel = self.algae_vel.copy()

        # Parameters
        a_r = self.robot_acc  # Robot acceleration magnitude
        v_max = self.robot_max_vel  # Robot max velocity
        a_a = self.algae_decel  # Algae deceleration

        # First, determine if the algae will stop due to deceleration
        algae_speed = np.linalg.norm(algae_vel)

        # If algae has non-zero speed and positive deceleration
        if algae_speed > 0 and a_a > 0:
            # Time until algae stops
            t_stop_algae = algae_speed / a_a

            # Calculate algae position when it stops
            # For decelerated motion: s = v₀t - ½at²
            if algae_speed > 0:
                # Direction of algae motion
                algae_dir = algae_vel / algae_speed

                # Distance traveled before stopping
                stop_distance = algae_speed * t_stop_algae - 0.5 * a_a * t_stop_algae**2

                # Final algae position when stopped
                algae_final_pos = algae_pos + algae_dir * stop_distance
            else:
                # Algae already stopped
                algae_final_pos = algae_pos
        else:
            # If no deceleration or already stopped
            t_stop_algae = float("inf")
            algae_final_pos = None  # Will not stop

        # Try solving for interception after algae stops first (simpler case)
        if t_stop_algae < float("inf"):
            # Calculate time to reach the stationary algae position
            intercept_after_stop = self.calculate_time_to_position(
                robot_pos, robot_vel, algae_final_pos, a_r, v_max
            )

            if intercept_after_stop:
                intercept_time, intercept_point = intercept_after_stop

                # Check if interception happens after algae has stopped
                if intercept_time > t_stop_algae:
                    #  Check if intercept time is more than the set stop time
                    if intercept_time > STOP_TIME_THRESHOLD:
                        self.intercept_point = intercept_point
                        self.intercept_time = intercept_time
                        self.intercept_calculated = True
                        self.fixed_target = self.intercept_point.copy()
                        self.predicted_robot_path = self.calculate_predicted_robot_path(intercept_time)
                        self.paused = True  # Pause if intercept time > 10
                        return
                    else:
                        self.intercept_point = intercept_point
                        self.intercept_time = intercept_time
                        self.intercept_calculated = True
                        # Store direct path to intercept point as a fixed target
                        self.fixed_target = self.intercept_point.copy()
                        # Store the predicted path for navigation
                        self.predicted_robot_path = self.calculate_predicted_robot_path(intercept_time)
                        return

        # For interception before algae stops or with continuous motion,
        # First, check if interception is even possible:
        # If algae is faster than robot's max speed and moving away
        if algae_speed > v_max:
            # Check if algae is moving away from robot
            vec_to_algae = algae_pos - robot_pos
            if np.dot(vec_to_algae, algae_vel) > 0:  # Moving away
                # Only possible if algae decelerates
                if a_a <= 0:
                    self.intercept_point = None
                    self.intercept_time = None
                    self.intercept_calculated = False
                    self.fixed_target = None
                    self.predicted_robot_path = None
                    return

        # Use numerical solution with differential equations for the general case
        self.find_intercept_numerical(max_time=30.0)
        
        # Check if intercept time is more than 10 seconds
        if self.intercept_time is not None and self.intercept_time > 10.0:
            self.paused = True  # Pause simulation if intercept time > 10
            
        # After calculating intercept_point, store it as the fixed target
        if self.intercept_point is not None:
            self.fixed_target = self.intercept_point.copy()
            # Generate the predicted robot path for accurate navigation
            self.predicted_robot_path = self.calculate_predicted_robot_path(self.intercept_time)
        else:
            self.fixed_target = None
            self.predicted_robot_path = None
            # Don't run the simulation if there's no intercept point
            self.paused = True
    """
    Calculate the predicted robot path to the interception point.
    This provides a trajectory for the robot to follow for precise interception.

    Args:
        intercept_time (float): The predicted time of interception

    Returns:
        list: List of predicted robot positions at each time step
    """
    def calculate_predicted_robot_path(self, intercept_time):
        
        if self.intercept_point is None or intercept_time is None:
            return None
            
        # Initial state
        predicted_path = []
        robot_pos = self.robot_pos.copy()
        robot_vel = self.robot_vel.copy()
        target_pos = self.intercept_point.copy()
        
        # Calculate time steps
        steps = int(intercept_time / self.dt)
        time_remaining = intercept_time
        
        # Generate path points
        for _ in range(steps):
            predicted_path.append(robot_pos.copy())
            
            # Calculate optimal direction and acceleration for precise arrival
            time_remaining -= self.dt
            if time_remaining <= 0:
                break
                
            # Vector to target
            dir_to_target = target_pos - robot_pos
            distance = np.linalg.norm(dir_to_target)
            
            if distance < 1e-6:  # Already at target
                continue
                
            dir_to_target = dir_to_target / distance
            
            # Calculate optimal acceleration using time-to-go guidance
            # This uses a proportional navigation approach adjusted for remaining time
            # v = d/t where d is remaining distance and t is remaining time
            desired_speed = distance / max(time_remaining, self.dt)
            desired_speed = min(desired_speed, self.robot_max_vel)
            
            # Desired velocity vector
            desired_vel = dir_to_target * desired_speed
            
            # Calculate acceleration needed
            accel_vec = (desired_vel - robot_vel) / self.dt
            accel_mag = np.linalg.norm(accel_vec)
            
            # Limit acceleration to maximum
            if accel_mag > self.robot_acc:
                accel_vec = accel_vec * (self.robot_acc / accel_mag)
                
            # Apply acceleration
            robot_vel += accel_vec * self.dt
            
            # Limit to max velocity
            speed = np.linalg.norm(robot_vel)
            if speed > self.robot_max_vel:
                robot_vel = (robot_vel / speed) * self.robot_max_vel
                
            # Update position
            robot_pos += robot_vel * self.dt
            
        return predicted_path

    """
    Calculate the time required for the robot to reach a specific position
    considering acceleration constraints and max velocity.

    Args:
        start_pos (numpy.ndarray): Starting position of the robot
        start_vel (numpy.ndarray): Starting velocity of the robot
        target_pos (numpy.ndarray): Target position to reach
        acceleration (float): Robot acceleration magnitude
        max_vel (float): Robot maximum velocity constraint

    Returns:
        tuple or None: (time, intercept_point) if interception is possible, None otherwise.
    """
    def calculate_time_to_position(
        self, start_pos, start_vel, target_pos, acceleration, max_vel
    ):
        """
        Calculate the time required for the robot to reach a specific position
        considering acceleration constraints and max velocity.

        Returns (time, intercept_point) if interception is possible, None otherwise.
        """
        # Vector from start to target
        displacement = target_pos - start_pos
        distance = np.linalg.norm(displacement)

        # If we're already at the target
        if distance < 1e-6:
            return (0.0, start_pos.copy())

        # Direction to target
        direction = displacement / distance

        # Current velocity in the direction of the target
        vel_in_direction = np.dot(start_vel, direction)

        """
        Solve for time considering:
            1. Deceleration if moving away from target
            2. Acceleration up to max_vel
            3. Constant velocity travel
        """

        # Calculate time to come to a stop if moving away
        t_stop = 0
        stop_distance = 0
        new_start_pos = start_pos.copy()

        if vel_in_direction < 0:  # Moving away from target
            # Time to stop
            t_stop = abs(vel_in_direction) / acceleration

            # Distance covered while stopping
            stop_distance = abs(vel_in_direction * t_stop / 2)

            # New starting position after stopping
            new_start_pos = start_pos + direction * (-stop_distance)

            # Recalculate displacement and distance
            displacement = target_pos - new_start_pos
            distance = np.linalg.norm(displacement)

            # Update direction if needed
            if distance > 1e-6:
                direction = displacement / distance

        # Time to reach max velocity
        t_max = max_vel / acceleration

        # Distance covered during acceleration phase
        accel_distance = 0.5 * acceleration * t_max * t_max

        # Calculate total time
        if accel_distance >= distance:
            # Target reached during acceleration phase
            t_accel = math.sqrt(2 * distance / acceleration)
            total_time = t_stop + t_accel
            return (total_time, target_pos.copy())
        else:
            # Need to travel at max velocity for some time
            remaining_distance = distance - accel_distance
            t_const = remaining_distance / max_vel
            total_time = t_stop + t_max + t_const
            return (total_time, target_pos.copy())

    """
    Find interception using numerical integration of the motion equations.
    Uses the same time step as the main simulation for consistency.

    Args:
        max_time (float): Maximum simulation time to search for interception.
                        Default is 30.0 seconds.

    Updates the intercept_point, intercept_time, and intercept_calculated attributes
    with the numerical solution results.
    """
    def find_intercept_numerical(self, max_time=30.0):
        # Initial states - use float64 for better precision
        robot_pos = self.robot_pos.copy().astype(np.float64)
        robot_vel = self.robot_vel.copy().astype(np.float64)
        algae_pos = self.algae_pos.copy().astype(np.float64)
        algae_vel = self.algae_vel.copy().astype(np.float64)
        time_step = self.dt

        # For storing minimum distance and its corresponding time
        min_distance = float("inf")
        min_distance_time = 0
        min_distance_positions = (None, None)

        # Store predicted paths (optionally for visualization)
        predicted_robot_path = [robot_pos.copy()]
        predicted_algae_path = [algae_pos.copy()]

        # Simulate motion over time using the exact same physics as the main simulation
        t = 0
        while t < max_time:
            # Current distance between robot and algae
            current_distance = np.linalg.norm(robot_pos - algae_pos)

            # Check if this is the closest they've been
            if current_distance < min_distance:
                min_distance = current_distance
                min_distance_time = t
                min_distance_positions = (robot_pos.copy(), algae_pos.copy())

            # Check for interception - use the same threshold as in simulate_step
            if current_distance < COLLISION_THRESHOLD:  # Threshold for interception
                self.intercept_point = (
                    robot_pos.copy() + algae_pos.copy()
                ) / 2  # Midpoint between them
                self.intercept_time = t
                self.intercept_calculated = True
                return

            # Calculate direction to algae
            dir_to_algae = algae_pos - robot_pos
            if np.linalg.norm(dir_to_algae) > 0:
                dir_to_algae = dir_to_algae / np.linalg.norm(dir_to_algae)
            else:
                dir_to_algae = np.array([0.0, 0.0])

            # Update robot velocity with acceleration toward algae
            robot_vel += dir_to_algae * self.robot_acc * time_step

            # Enforce max velocity constraint
            speed = np.linalg.norm(robot_vel)
            if speed > self.robot_max_vel:
                robot_vel = (robot_vel / speed) * self.robot_max_vel

            # Update robot position
            robot_pos += robot_vel * time_step

            # Update algae velocity (with deceleration if applicable)
            algae_speed = np.linalg.norm(algae_vel)
            if algae_speed > 0 and self.algae_decel > 0:
                # Direction of algae motion
                dir_algae = algae_vel / algae_speed

                # Apply deceleration (capped to prevent velocity reversal)
                decel_magnitude = min(self.algae_decel * time_step, algae_speed)
                algae_vel -= dir_algae * decel_magnitude

            # Update algae position
            algae_pos += algae_vel * time_step

            # Store paths (optional)
            predicted_robot_path.append(robot_pos.copy())
            predicted_algae_path.append(algae_pos.copy())

            # Advance time
            t += time_step

        """
        If we get here, no direct interception was found within max_time
        Only return a closest approach if it's within a reasonable threshold of actual interception
        """

        if (
            min_distance < COLLISION_THRESHOLD * 1.5
        ):  # Use a slightly larger threshold for closest approach
            self.intercept_point = (
                min_distance_positions[0] + min_distance_positions[1]
            ) / 2
            self.intercept_time = min_distance_time
            self.intercept_calculated = True
        else:
            self.intercept_point = None
            self.intercept_time = None
            self.intercept_calculated = False

    """
    Update positions and velocities for one time step of the simulation.
    Applies physics rules including acceleration toward target, velocity constraints,
    deceleration, and collision detection.
    """
    def simulate_step(self):
        # Don't proceed if paused or if there's no interception possible
        if self.paused or self.fixed_target is None:
            return

        # Calculate time elapsed in the simulation
        elapsed_time = self.time
        
        # Check if we have a predicted path to follow
        if hasattr(self, 'predicted_robot_path') and self.predicted_robot_path:
            # Find the closest timepoint in our predicted path
            time_index = int(elapsed_time / self.dt)
            if time_index < len(self.predicted_robot_path):
                # Use the predicted position for this time step as our target
                current_target = self.predicted_robot_path[time_index]
                
                # Calculate direction to the current waypoint
                dir_to_target = current_target - self.robot_pos
                distance_to_waypoint = np.linalg.norm(dir_to_target)
                
                if distance_to_waypoint > 0:
                    dir_to_target = dir_to_target / distance_to_waypoint
                else:
                    # If we're at the waypoint, aim directly for the final target
                    dir_to_target = self.fixed_target - self.robot_pos
                    if np.linalg.norm(dir_to_target) > 0:
                        dir_to_target = dir_to_target / np.linalg.norm(dir_to_target)
                    else:
                        dir_to_target = np.array([0.0, 0.0])
                        
                # Calculate time remaining until interception
                time_remaining = max(0.001, self.intercept_time - elapsed_time)
                
                # Calculate optimal acceleration with proportional navigation
                # Adjust velocity based on remaining distance and time
                remaining_distance = np.linalg.norm(self.fixed_target - self.robot_pos)
                desired_speed = min(remaining_distance / time_remaining, self.robot_max_vel)
                
                # Current velocity in the desired direction
                current_vel_magnitude = np.dot(self.robot_vel, dir_to_target)
                
                # Determine if we need to accelerate or decelerate
                if current_vel_magnitude < desired_speed:
                    # Need to accelerate
                    self.robot_vel += dir_to_target * self.robot_acc * self.dt
                else:
                    # Need to decelerate or maintain speed
                    # Calculate deceleration needed
                    self.robot_vel = dir_to_target * desired_speed
            else:
                # We've run past our predicted path, aim directly at the target
                dir_to_target = self.fixed_target - self.robot_pos
                if np.linalg.norm(dir_to_target) > 0:
                    dir_to_target = dir_to_target / np.linalg.norm(dir_to_target)
                else:
                    dir_to_target = np.array([0.0, 0.0])
                
                # Apply acceleration in direction of target
                self.robot_vel += dir_to_target * self.robot_acc * self.dt
        else:
            # If no predicted path exists, fall back to direct navigation
            dir_to_target = self.fixed_target - self.robot_pos
            if np.linalg.norm(dir_to_target) > 0:
                dir_to_target = dir_to_target / np.linalg.norm(dir_to_target)
            else:
                dir_to_target = np.array([0.0, 0.0])
            
            # Apply acceleration in direction of target
            self.robot_vel += dir_to_target * self.robot_acc * self.dt

        # Limit to max velocity
        speed = np.linalg.norm(self.robot_vel)
        if speed > self.robot_max_vel:
            self.robot_vel = (self.robot_vel / speed) * self.robot_max_vel

        # Update robot position
        self.robot_pos += self.robot_vel * self.dt

        # Update algae velocity (with deceleration if applicable)
        if np.linalg.norm(self.algae_vel) > 0 and self.algae_decel > 0:
            # Direction of algae motion
            dir_algae = self.algae_vel / np.linalg.norm(self.algae_vel)

            # Apply deceleration
            decel_magnitude = min(
                self.algae_decel * self.dt, np.linalg.norm(self.algae_vel)
            )
            self.algae_vel -= dir_algae * decel_magnitude

        # Update algae position
        self.algae_pos += self.algae_vel * self.dt

        # Track paths
        self.path_points_robot.append(self.robot_pos.copy())
        self.path_points_algae.append(self.algae_pos.copy())

        # Update time
        self.time += self.dt

        # Check if interception occurred
        distance = np.linalg.norm(self.robot_pos - self.algae_pos)
        if distance < COLLISION_THRESHOLD:  # Threshold for interception
            print(f"Interception occurred at time {self.time:.2f}s")
            print(
                f"Predicted interception time: {self.intercept_time:.2f}s"
                if self.intercept_time
                else "No prediction"
            )
            self.paused = True

    """
    Render the current simulation state to the screen.
    Draws the grid, paths, robot, algae, velocity vectors, predicted intercept point,
    and all text information including parameters and controls.
    """
    def draw(self):
        # Draw the simulation state

        self.screen.fill(BG_COLOR)

        # Draw grid
        grid_spacing = 50
        for x in range(0, WIDTH, grid_spacing):
            pygame.draw.line(self.screen, LINE_COLOR, (x, 0), (x, HEIGHT), 1)
        for y in range(0, HEIGHT, grid_spacing):
            pygame.draw.line(self.screen, LINE_COLOR, (0, y), (WIDTH, y), 1)

        # Draw paths if not in setup mode
        if not (self.playground_mode and not self.setup_completed):
            if len(self.path_points_robot) > 1:
                pygame.draw.lines(
                    self.screen,
                    ROBOT_COLOR,
                    False,
                    [(p[0], p[1]) for p in self.path_points_robot],
                    2,
                )
            if len(self.path_points_algae) > 1:
                pygame.draw.lines(
                    self.screen,
                    ALGAE_COLOR,
                    False,
                    [(p[0], p[1]) for p in self.path_points_algae],
                    2,
                )

        # Draw predicted interception point
        if self.intercept_point is not None:
            pygame.draw.circle(
                self.screen,
                PREDICTION_COLOR,
                (int(self.intercept_point[0]), int(self.intercept_point[1])),
                10,
                2,
            )
            # Add interception time text near the predicted point in playground mode
            if self.intercept_time and self.playground_mode and self.setup_stage == 5:
                self.draw_text(
                    f"Intercept at {self.intercept_time:.2f}s",
                    int(self.intercept_point[0]) + 15,
                    int(self.intercept_point[1]) - 15,
                    PREDICTION_COLOR,
                )

        # Draw robot and algae
        pygame.draw.circle(
            self.screen,
            ROBOT_COLOR,
            (int(self.robot_pos[0]), int(self.robot_pos[1])),
            15,
        )
        pygame.draw.circle(
            self.screen,
            ALGAE_COLOR,
            (int(self.algae_pos[0]), int(self.algae_pos[1])),
            15,
        )

        # Draw velocity vectors
        self.draw_vector(
            self.robot_pos,
            self.robot_vel,
            ROBOT_COLOR,
            highlighted=hasattr(self, "editing_robot_vel") and self.editing_robot_vel,
        )
        self.draw_vector(
            self.algae_pos,
            self.algae_vel,
            ALGAE_COLOR,
            highlighted=hasattr(self, "editing_algae_vel") and self.editing_algae_vel,
        )

        # Draw temporary velocity line in playground mode
        if self.setting_robot_vel or self.setting_algae_vel:
            mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float64)
            pygame.draw.line(
                self.screen,
                (255, 255, 255),
                (int(self.vel_start_pos[0]), int(self.vel_start_pos[1])),
                (int(mouse_pos[0]), int(mouse_pos[1])),
                2,
            )
            # Draw preview velocity arrow
            dir_vec = mouse_pos - self.vel_start_pos
            speed = (
                np.linalg.norm(dir_vec) * 0.5
            )  # Same scale factor as in handle_playground_events
            if np.linalg.norm(dir_vec) > 0:
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
                self.draw_text(
                    f"Speed: {speed:.1f}",
                    int(mouse_pos[0]) + 15,
                    int(mouse_pos[1]),
                    (255, 255, 255),
                )

        # Highlight object being edited
        if hasattr(self, "editing_robot") and self.editing_robot:
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                (int(self.robot_pos[0]), int(self.robot_pos[1])),
                20,
                2,
            )
        if hasattr(self, "editing_algae") and self.editing_algae:
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                (int(self.algae_pos[0]), int(self.algae_pos[1])),
                20,
                2,
            )

        # Draw text information
        self.draw_text(f"Time: {self.time:.2f}s", 20, 20)
        self.draw_text(f"Robot Velocity: {np.linalg.norm(self.robot_vel):.2f}", 20, 40)
        self.draw_text(f"Algae Velocity: {np.linalg.norm(self.algae_vel):.2f}", 20, 60)
        self.draw_text(f"Robot Acceleration: {self.robot_acc:.1f}", 20, 80)
        self.draw_text(f"Robot Max Velocity: {self.robot_max_vel:.1f}", 20, 100)
        self.draw_text(f"Algae Deceleration: {self.algae_decel:.1f}", 20, 120)

        if self.intercept_time and not self.playground_mode:
            self.draw_text(
                f"Predicted Interception Time: {self.intercept_time:.2f}s", 20, 140
            )
        elif not self.playground_mode:
            self.draw_text("No interception predicted", 20, 140)

        if self.paused and not self.playground_mode:
            self.draw_text(
                "PAUSED (Space to continue, R to reset)", WIDTH // 2, 30, centered=True
            )

        # Draw current case
        self.draw_text(
            f"Case: {self.current_case.value}",
            WIDTH // 2,
            HEIGHT - 30,
            centered=True,
            font=self.large_font,
        )

        if self.playground_mode and self.setup_stage == 5:
            if hasattr(self, "editing_robot") and self.editing_robot:
                self.draw_text(
                    "Editing: Robot Position",
                    WIDTH // 2,
                    70,
                    centered=True,
                    color=(255, 200, 100),
                )
            elif hasattr(self, "editing_algae") and self.editing_algae:
                self.draw_text(
                    "Editing: Algae Position",
                    WIDTH // 2,
                    70,
                    centered=True,
                    color=(255, 200, 100),
                )
            elif hasattr(self, "editing_robot_vel") and self.editing_robot_vel:
                self.draw_text(
                    "Editing: Robot Velocity",
                    WIDTH // 2,
                    70,
                    centered=True,
                    color=(255, 200, 100),
                )
            elif hasattr(self, "editing_algae_vel") and self.editing_algae_vel:
                self.draw_text(
                    "Editing: Algae Velocity",
                    WIDTH // 2,
                    70,
                    centered=True,
                    color=(255, 200, 100),
                )

        # Add new information display for playground mode
        if self.playground_mode and self.setup_stage == 5 and not self.setup_completed:
            # Display intercept information in a box
            infobox_height = 100
            infobox_width = 400
            pygame.draw.rect(
                self.screen,
                (0, 0, 0, 128),
                (
                    WIDTH // 2 - infobox_width // 2,
                    HEIGHT - infobox_height - 70,
                    infobox_width,
                    infobox_height,
                ),
                0,
            )
            pygame.draw.rect(
                self.screen,
                (200, 200, 200),
                (
                    WIDTH // 2 - infobox_width // 2,
                    HEIGHT - infobox_height - 70,
                    infobox_width,
                    infobox_height,
                ),
                1,
            )

            # Add editing instructions
            self.draw_text(
                "Right-click robot or algae to move them",
                WIDTH // 2,
                HEIGHT - infobox_height - 60,
                centered=True,
                color=(220, 220, 220),
            )
            self.draw_text(
                "BACKSPACE: Restore previous state",
                WIDTH // 2,
                HEIGHT - infobox_height - 40,
                centered=True,
                color=(220, 220, 220),
            )

            # Show interception information
            if self.intercept_point is not None:
                self.draw_text(
                    f"Interception predicted at time: {self.intercept_time:.2f}s",
                    WIDTH // 2,
                    HEIGHT - infobox_height - 10,
                    centered=True,
                    color=(100, 255, 100),
                )
            else:
                self.draw_text(
                    "No interception predicted with current settings",
                    WIDTH // 2,
                    HEIGHT - infobox_height - 10,
                    centered=True,
                    color=(255, 100, 100),
                )
        # Draw appropriate controls based on mode
        if self.playground_mode and not self.setup_completed:
            if self.setup_stage == 1:
                self.draw_text(
                    "Click to place the robot",
                    WIDTH // 2,
                    30,
                    centered=True,
                    font=self.large_font,
                )
            elif self.setup_stage == 2:
                self.draw_text(
                    "Click to set robot velocity (direction and magnitude)",
                    WIDTH // 2,
                    30,
                    centered=True,
                    font=self.large_font,
                )
            elif self.setup_stage == 3:
                self.draw_text(
                    "Click to place the algae",
                    WIDTH // 2,
                    30,
                    centered=True,
                    font=self.large_font,
                )
            elif self.setup_stage == 4:
                self.draw_text(
                    "Click to set algae velocity (direction and magnitude)",
                    WIDTH // 2,
                    30,
                    centered=True,
                    font=self.large_font,
                )
            elif self.setup_stage == 5:
                instructions = [
                    "Adjust parameters:",
                    "UP/DOWN: Robot acceleration",
                    "LEFT/RIGHT: Robot max velocity",
                    "A/Z: Algae deceleration",
                    "ENTER: Start simulation",
                    "ESC: Cancel and return to normal mode",
                ]
                for i, line in enumerate(instructions):
                    self.draw_text(line, WIDTH // 2, 30 + i * 20, centered=True)

            self.draw_text(
                f"Setup Stage: {self.setup_stage}/5",
                WIDTH // 2,
                HEIGHT - 60,
                centered=True,
            )
        else:
            controls = "Controls: 1-6 for preset cases, 7 for playground mode, Space to pause/resume, R to reset"
            self.draw_text(controls, WIDTH // 2, HEIGHT - 60, centered=True)

        pygame.display.flip()

    """
    Draw a velocity vector with an arrowhead starting at the specified position.

    Args:
        pos (numpy.ndarray): Starting position of the vector
        vel (numpy.ndarray): Velocity vector to draw
        color (tuple): RGB color for the vector
        scale (float): Scale factor to visualize the vector. Default is 0.5
        highlighted (bool): Whether to draw the vector highlighted. Default is False
    """
    def draw_vector(self, pos, vel, color, scale=0.5, highlighted=False):
        # Draw a velocity vector starting at position
        if np.linalg.norm(vel) > 0:
            end_pos = pos + vel * scale

            # Draw thicker line or different color if vector is being edited
            if highlighted:
                # Draw a glow effect with larger, semi-transparent line
                glow_color = (255, 255, 255, 128)  # White glow
                pygame.draw.line(
                    self.screen,
                    glow_color,
                    (int(pos[0]), int(pos[1])),
                    (int(end_pos[0]), int(end_pos[1])),
                    6,
                )
                # Draw the vector in original color but thicker
                pygame.draw.line(
                    self.screen,
                    color,
                    (int(pos[0]), int(pos[1])),
                    (int(end_pos[0]), int(end_pos[1])),
                    3,
                )
            else:
                # Normal drawing
                pygame.draw.line(
                    self.screen,
                    color,
                    (int(pos[0]), int(pos[1])),
                    (int(end_pos[0]), int(end_pos[1])),
                    2,
                )

            # Draw arrowhead
            angle = math.atan2(end_pos[1] - pos[1], end_pos[0] - pos[0])
            arrowhead_points = [
                (int(end_pos[0]), int(end_pos[1])),
                (
                    int(end_pos[0] - 10 * math.cos(angle - math.pi / 6)),
                    int(end_pos[1] - 10 * math.sin(angle - math.pi / 6)),
                ),
                (
                    int(end_pos[0] - 10 * math.cos(angle + math.pi / 6)),
                    int(end_pos[1] - 10 * math.sin(angle + math.pi / 6)),
                ),
            ]

            # Draw larger arrowhead if highlighted
            if highlighted:
                pygame.draw.polygon(self.screen, color, arrowhead_points, 0)  # Filled
                # Draw circle at the end for easier grabbing
                pygame.draw.circle(
                    self.screen,
                    (255, 255, 255),
                    (int(end_pos[0]), int(end_pos[1])),
                    8,
                    2,
                )
            else:
                pygame.draw.polygon(self.screen, color, arrowhead_points, 0)

    """
    Draw text on the screen with specified parameters.

    Args:
        text (str): Text to display
        x (int): X-coordinate for text position
        y (int): Y-coordinate for text position
        color (tuple): RGB color for the text. Default is TEXT_COLOR
        centered (bool): If True, coordinates represent the center of the text.
                        If False, coordinates represent the top-left corner.
                        Default is False
        font (pygame.font.Font): Font to use. Default is self.font
    """
    def draw_text(self, text, x, y, color=TEXT_COLOR, centered=False, font=None):
        # Draw text on screen
        if font is None:
            font = self.font
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if centered:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        self.screen.blit(text_surface, text_rect)

    """
    Handle user input events specific to playground mode.
    Manages object placement, velocity setting, parameter adjustments,
    and direct manipulation of simulation objects.

    Args:
        event (pygame.event.Event): The pygame event to process
    """
    def handle_playground_events(self, event):
        # Handle events specific to playground mode
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float64)

                if self.placing_robot:
                    # Place robot
                    self.robot_pos = mouse_pos
                    self.placing_robot = False
                    self.setting_robot_vel = True
                    self.vel_start_pos = mouse_pos.copy()
                    self.setup_stage = 2

                elif self.setting_robot_vel:
                    # Set robot velocity
                    end_pos = mouse_pos
                    self.robot_vel = (
                        end_pos - self.vel_start_pos
                    ) * 0.5  # Scale for better control
                    self.setting_robot_vel = False
                    self.placing_algae = True
                    self.setup_stage = 3

                elif self.placing_algae:
                    # Place algae
                    self.algae_pos = mouse_pos
                    self.placing_algae = False
                    self.setting_algae_vel = True
                    self.vel_start_pos = mouse_pos.copy()
                    self.setup_stage = 4

                elif self.setting_algae_vel:
                    # Set algae velocity
                    end_pos = mouse_pos
                    self.algae_vel = (
                        end_pos - self.vel_start_pos
                    ) * 0.5  # Scale for better control
                    self.setting_algae_vel = False
                    self.setup_stage = 5  # Final setup stage: parameters
                    # Calculate and show interception prediction
                    self.calculate_intercept()

            # Right-click to select and edit positions or vectors in stage 5
            elif event.button == 3 and self.setup_stage == 5:
                mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float64)

                # Calculate vector endpoints for detection
                robot_vel_endpoint = self.robot_pos + self.robot_vel * 0.5
                algae_vel_endpoint = self.algae_pos + self.algae_vel * 0.5

                # Store initial positions for offset calculation
                self.initial_mouse_pos = mouse_pos.copy()

                # Check if clicking near the velocity vector endpoints first (prioritize vectors)
                if np.linalg.norm(mouse_pos - robot_vel_endpoint) < 20:
                    self.editing_robot = False
                    self.editing_algae = False
                    self.editing_robot_vel = True
                    self.editing_algae_vel = False
                    self.initial_robot_vel = self.robot_vel.copy()
                    print("Editing robot velocity vector")
                elif np.linalg.norm(mouse_pos - algae_vel_endpoint) < 20:
                    self.editing_robot = False
                    self.editing_algae = False
                    self.editing_robot_vel = False
                    self.editing_algae_vel = True
                    self.initial_algae_vel = self.algae_vel.copy()
                    print("Editing algae velocity vector")
                # Check if clicking near robot or algae
                elif np.linalg.norm(mouse_pos - self.robot_pos) < 20:  # Near robot
                    self.editing_robot = True
                    self.editing_algae = False
                    self.editing_robot_vel = True  # Now also edit velocity
                    self.editing_algae_vel = False
                    self.initial_robot_pos = self.robot_pos.copy()
                    self.initial_robot_vel = self.robot_vel.copy()
                    print("Editing robot position and velocity")
                elif np.linalg.norm(mouse_pos - self.algae_pos) < 20:  # Near algae
                    self.editing_robot = False
                    self.editing_algae = True
                    self.editing_robot_vel = False
                    self.editing_algae_vel = True  # Now also edit velocity
                    self.initial_algae_pos = self.algae_pos.copy()
                    self.initial_algae_vel = self.algae_vel.copy()
                    print("Editing algae position and velocity")
                else:
                    self.editing_robot = False
                    self.editing_algae = False
                    self.editing_robot_vel = False
                    self.editing_algae_vel = False

        elif event.type == pygame.MOUSEMOTION:
            mouse_pos = np.array(pygame.mouse.get_pos(), dtype=np.float64)

            # Calculate mouse movement delta from initial click
            if hasattr(self, "initial_mouse_pos"):
                delta = mouse_pos - self.initial_mouse_pos

                # Move objects being edited
                if hasattr(self, "editing_robot") and self.editing_robot:
                    if hasattr(self, "initial_robot_pos"):
                        self.robot_pos = self.initial_robot_pos + delta
                    else:
                        self.robot_pos = mouse_pos

                    # If also editing velocity, maintain the same velocity direction and magnitude
                    if (
                        hasattr(self, "editing_robot_vel")
                        and self.editing_robot_vel
                        and hasattr(self, "initial_robot_vel")
                    ):
                        self.robot_vel = self.initial_robot_vel.copy()

                    self.calculate_intercept()  # Update interception prediction

                elif hasattr(self, "editing_algae") and self.editing_algae:
                    if hasattr(self, "initial_algae_pos"):
                        self.algae_pos = self.initial_algae_pos + delta
                    else:
                        self.algae_pos = mouse_pos

                    # If also editing velocity, maintain the same velocity direction and magnitude
                    if (
                        hasattr(self, "editing_algae_vel")
                        and self.editing_algae_vel
                        and hasattr(self, "initial_algae_vel")
                    ):
                        self.algae_vel = self.initial_algae_vel.copy()

                    self.calculate_intercept()  # Update interception prediction

                # Edit velocity vectors (but only directly if not also moving position)
                elif hasattr(self, "editing_robot_vel") and self.editing_robot_vel:
                    # Update robot velocity vector - pointing from robot to mouse
                    self.robot_vel = (
                        mouse_pos - self.robot_pos
                    ) * 0.5  # Scale for better control
                    self.calculate_intercept()  # Update interception prediction

                elif hasattr(self, "editing_algae_vel") and self.editing_algae_vel:
                    # Update algae velocity vector - pointing from algae to mouse
                    self.algae_vel = (
                        mouse_pos - self.algae_pos
                    ) * 0.5  # Scale for better control
                    self.calculate_intercept()  # Update interception prediction

        elif event.type == pygame.MOUSEBUTTONUP:
            # Stop editing positions and vectors
            self.editing_robot = False
            self.editing_algae = False
            self.editing_robot_vel = False
            self.editing_algae_vel = False
            # Clean up initial positions
            if hasattr(self, "initial_mouse_pos"):
                del self.initial_mouse_pos
            if hasattr(self, "initial_robot_pos"):
                del self.initial_robot_pos
            if hasattr(self, "initial_algae_pos"):
                del self.initial_algae_pos
            if hasattr(self, "initial_robot_vel"):
                del self.initial_robot_vel
            if hasattr(self, "initial_algae_vel"):
                del self.initial_algae_vel

        elif event.type == pygame.KEYDOWN and self.setup_stage == 5:
            if event.key == pygame.K_UP:
                # Increase robot acceleration
                self.robot_acc = min(self.robot_acc + 10, 300)
                self.calculate_intercept()
            elif event.key == pygame.K_DOWN:
                # Decrease robot acceleration
                self.robot_acc = max(self.robot_acc - 10, 10)
                self.calculate_intercept()
            elif event.key == pygame.K_RIGHT:
                # Increase robot max velocity
                self.robot_max_vel = min(self.robot_max_vel + 10, 400)
                self.calculate_intercept()
            elif event.key == pygame.K_LEFT:
                # Decrease robot max velocity
                self.robot_max_vel = max(self.robot_max_vel - 10, 50)
                self.calculate_intercept()
            elif event.key == pygame.K_a:
                # Increase algae deceleration
                self.algae_decel = min(self.algae_decel + 5, 100)
                self.calculate_intercept()
            elif event.key == pygame.K_z:
                # Decrease algae deceleration
                self.algae_decel = max(self.algae_decel - 5, 0)
                self.calculate_intercept()
            elif event.key == pygame.K_BACKSPACE:
                # Restore previous state (soft reset)
                self.restore_previous_state()
            #  W key resets positions to starting points in playground mode
            elif event.key == pygame.K_w:
                # Store current velocities
                robot_vel_temp = self.robot_vel.copy()
                algae_vel_temp = self.algae_vel.copy()
                
                # Reset positions to initial positions from when playground mode was entered
                if hasattr(self, "initial_setup_robot_pos"):
                    self.robot_pos = self.initial_setup_robot_pos.copy()
                else:
                    # If no initial positions stored, save current ones for future resets
                    self.initial_setup_robot_pos = self.robot_pos.copy()
                    
                if hasattr(self, "initial_setup_algae_pos"):
                    self.algae_pos = self.initial_setup_algae_pos.copy()
                else:
                    # If no initial positions stored, save current ones for future resets
                    self.initial_setup_algae_pos = self.algae_pos.copy()
                    
                # Keep the velocities the same
                self.robot_vel = robot_vel_temp
                self.algae_vel = algae_vel_temp
                
                # Reset paths
                self.path_points_robot = [self.robot_pos.copy()]
                self.path_points_algae = [self.algae_pos.copy()]
                
                # Reset time
                self.time = 0.0
                
                # Recalculate interception
                self.calculate_intercept()
                
            elif event.key == pygame.K_RETURN:
                # Start simulation
                self.setup_completed = True
                self.playground_mode = False
                self.paused = False
                self.path_points_robot = [self.robot_pos.copy()]
                self.path_points_algae = [self.algae_pos.copy()]
                
                # Store initial positions for W key reset functionality
                self.initial_setup_robot_pos = self.robot_pos.copy()
                self.initial_setup_algae_pos = self.algae_pos.copy()
                
                self.calculate_intercept()
    """
    Main simulation loop. Handles events, updates simulation state,
    renders graphics, and maintains timing.
    """
    def run(self):
        # Main simulation loop
        self.paused = False
        # Setting default values for editing flags
        self.editing_robot = False
        self.editing_algae = False

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if self.playground_mode and not self.setup_completed:
                        if event.key == pygame.K_ESCAPE:
                            # Exit playground mode
                            self.reset_simulation(SimulationCase.NORMAL)
                        else:
                            # Handle playground-specific key events
                            self.handle_playground_events(event)
                    else:
                        # Regular simulation controls
                        if event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                        elif event.key == pygame.K_r:
                            self.reset_simulation(self.current_case)
                        elif event.key == pygame.K_BACKSPACE:
                            # Add soft reset for non-playground mode too
                            self.restore_previous_state()
                        elif event.key == pygame.K_1:
                            self.reset_simulation(SimulationCase.NORMAL)
                        elif event.key == pygame.K_2:
                            self.reset_simulation(SimulationCase.ALGAE_RESTS)
                        elif event.key == pygame.K_3:
                            self.reset_simulation(SimulationCase.NO_MAX_VELOCITY)
                        elif event.key == pygame.K_4:
                            self.reset_simulation(SimulationCase.ALGAE_IN_FRONT)
                        elif event.key == pygame.K_5:
                            self.reset_simulation(SimulationCase.FAST_ALGAE)
                        elif event.key == pygame.K_6:
                            self.reset_simulation(SimulationCase.NEGATIVE_INITIAL)
                        elif event.key == pygame.K_7:
                            self.reset_simulation(SimulationCase.PLAYGROUND)

                # Handle mouse events for position editing with mousemotion and mouseup too
                if self.playground_mode and not self.setup_completed:
                    self.handle_playground_events(event)

            if not (self.playground_mode and not self.setup_completed) and self.fixed_target is not None:
                self.simulate_step()
            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


"""
Initialize the simulator with default settings, create the display window,
and set up initial simulation parameters.
"""
if __name__ == "__main__":
    simulator = Simulator()
    print("Robot-Algae Interception Simulator")
    print("----------------------------------")
    print("Controls:")
    print("1-6: Select predefined test cases")
    print("7: Enter playground mode (custom scenario)")
    print("Space: Pause/resume simulation")
    print("R: Reset current case")
    print("ESC: Exit playground mode")
    print("\nPlayground Mode Controls:")
    print("- Click to place robot")
    print("- Click again to set robot velocity")
    print("- Click to place algae")
    print("- Click again to set algae velocity")
    print("- Use arrow keys to adjust robot parameters")
    print("- Use A/Z to adjust algae deceleration")
    print("- Press ENTER to start simulation")
    simulator.run()