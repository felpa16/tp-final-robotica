import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import numpy as np
from scipy.spatial.transform import Rotation as R
import heapq
from enum import Enum

from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, PoseArray, TransformStamped, Quaternion, PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformBroadcaster, TransformListener, Buffer

class State(Enum):
    IDLE = 0
    PLANNING = 1
    NAVIGATING = 2
    AVOIDING_OBSTACLE = 3


class AmclNode(Node):
    def __init__(self):
        super().__init__('my_py_amcl')

        # --- Parameters ---
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_footprint')
        self.declare_parameter('map_frame_id', 'map')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('initial_pose_topic', 'initialpose')
        self.declare_parameter('laser_max_range', 3.5)
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('obstacle_detection_distance', 0.3)
        self.declare_parameter('obstacle_avoidance_turn_speed', 0.5)


        # --- Parameters to set ---
        # TODO: Setear valores default
        self.declare_parameter('num_particles', 10) #empezamos con pocas, de ultima subimos dsp
        self.declare_parameter('alpha1', 0.1)
        self.declare_parameter('alpha2', 0.1)
        self.declare_parameter('alpha3', 0.01)
        self.declare_parameter('alpha4', 0.01)
        self.declare_parameter('z_hit', 0.8)
        self.declare_parameter('z_rand', 0.2)
        self.declare_parameter('lookahead_distance', 1)
        self.declare_parameter('linear_velocity', 0.2)
        self.declare_parameter('goal_tolerance', 0.1)
        self.declare_parameter('path_pruning_distance', 0.2)
        self.declare_parameter('safety_margin_cells', 2)
        #FIN DEL TODO

        
        self.num_particles = self.get_parameter('num_particles').value
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value
        self.map_frame_id = self.get_parameter('map_frame_id').value
        self.laser_max_range = self.get_parameter('laser_max_range').value
        self.z_hit = self.get_parameter('z_hit').value
        self.z_rand = self.get_parameter('z_rand').value
        self.alphas = np.array([
            self.get_parameter('alpha1').value,
            self.get_parameter('alpha2').value,
            self.get_parameter('alpha3').value,
            self.get_parameter('alpha4').value,
        ])
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.linear_velocity = self.get_parameter('linear_velocity').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.path_pruning_distance = self.get_parameter('path_pruning_distance').value
        self.safety_margin_cells = self.get_parameter('safety_margin_cells').value
        self.obstacle_detection_distance = self.get_parameter('obstacle_detection_distance').value
        self.obstacle_avoidance_turn_speed = self.get_parameter('obstacle_avoidance_turn_speed').value

        # --- State ---
        self.particles = np.zeros((self.num_particles, 3))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.map_data = None
        self.latest_scan = None
        self.initial_pose_received = False
        self.map_received = False
        self.last_odom_pose = None
        self.state = State.IDLE
        self.current_path = None
        self.goal_pose = None
        self.inflated_grid = None
        self.obstacle_avoidance_start_yaw = None
        self.obstacle_avoidance_last_yaw = None
        self.obstacle_avoidance_cumulative_angle = 0.0
        self.obstacle_avoidance_active = False
        
        # --- ROS 2 Interfaces ---
        map_qos = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        scan_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        
        self.map_sub = self.create_subscription(OccupancyGrid, self.get_parameter('map_topic').value, self.map_callback, map_qos)
        self.scan_sub = self.create_subscription(LaserScan, self.get_parameter('scan_topic').value, self.scan_callback, scan_qos)
        self.initial_pose_sub = self.create_subscription(PoseWithCovarianceStamped, self.get_parameter('initial_pose_topic').value, self.initial_pose_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, self.get_parameter('goal_topic').value, self.goal_callback, 10)
        
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'amcl_pose', 10)
        self.particle_pub = self.create_publisher(MarkerArray, 'particle_cloud', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('MyPyAMCL node initialized.')
    
    #IMPLEMENTADO NUESTRO
    def inflate_map(self):
        if self.map_data is None:
            self.get_logger().warn("Cannot inflate map, map data is not available.")
            return
        
        # Create an inflated grid with a safety margin
        self.inflated_grid = np.copy(self.grid)
        kernel_size = 2 * self.safety_margin_cells + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        
        # Perform dilation to inflate the obstacles
        from scipy.ndimage import binary_dilation
        inflated_obstacles = binary_dilation(self.grid > 0, structure=kernel).astype(np.int8)
        
        # Update the inflated grid
        self.inflated_grid[inflated_obstacles > 0] = 100

    def map_callback(self, msg):
        if not self.map_received:
            self.map_data = msg
            self.map_received = True
            self.grid = np.array(self.map_data.data).reshape((self.map_data.info.height, self.map_data.info.width))
            self.inflate_map()
            self.get_logger().info('Map and inflated map processed.')

    def scan_callback(self, msg):
        self.latest_scan = msg

    def goal_callback(self, msg):
        if self.map_data is None:
            self.get_logger().warn("Goal received, but map is not available yet. Ignoring goal.")
            return

        if msg.header.frame_id != self.map_frame_id:
            self.get_logger().warn(f"Goal received in frame '{msg.header.frame_id}', but expected '{self.map_frame_id}'. Ignoring.")
            return
            
        self.goal_pose = msg.pose
        self.get_logger().info(f"New goal received: ({self.goal_pose.position.x:.2f}, {self.goal_pose.position.y:.2f}). State -> PLANNING")
        self.state = State.PLANNING
        self.current_path = None

    def initial_pose_callback(self, msg):
        if msg.header.frame_id != self.map_frame_id:
            self.get_logger().warn(f"Initial pose frame is '{msg.header.frame_id}' but expected '{self.map_frame_id}'. Ignoring.")
            return
        self.get_logger().info('Initial pose received.')
        self.initialize_particles(msg.pose.pose)
        self.initial_pose_received = True
        self.last_odom_pose = None # Reset odom tracking

    def initialize_particles(self, initial_pose):
        # TODO: Inicializar particulas en base a la pose inicial con variaciones aleatorias
        # Deben ser la misma cantidad de particulas que self.num_particles
        # Deben tener un peso
        
        x = initial_pose.position.x
        y = initial_pose.position.y
        q = initial_pose.orientation
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        
        position_std = 0.1  
        orientation_std = 0.1  
        
        self.particles[:, 0] = np.random.normal(x, position_std, self.num_particles)
        self.particles[:, 1] = np.random.normal(y, position_std, self.num_particles)
        self.particles[:, 2] = np.random.normal(yaw, orientation_std, self.num_particles)
        
        self.weights = np.ones(self.num_particles) / self.num_particles

        self.publish_particles()  

    def initialize_particles_randomly(self):
        # TODO: Inizializar particulas aleatoriamente en todo el mapa
        if self.map_data is None:
            self.get_logger().warn("Cannot initialize particles randomly, map data is not available.")
            return
        
        map_w = self.map_data.info.width
        map_h = self.map_data.info.height
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        
        free_cells = []
        for y in range(map_h):
            for x in range(map_w):
                if self.grid[y, x] <= 65:  # Free space si esta abajo del 0.65
                    world_x = x * resolution + origin_x
                    world_y = y * resolution + origin_y
                    free_cells.append((world_x, world_y))
        
        if len(free_cells) == 0:
            self.get_logger().warn("No free space found in map for particle initialization")
            return
            
        selected_indices = np.random.choice(len(free_cells), self.num_particles, replace=True)
        
        for i, idx in enumerate(selected_indices):
            self.particles[i, 0] = free_cells[idx][0]
            self.particles[i, 1] = free_cells[idx][1]
            self.particles[i, 2] = np.random.uniform(-np.pi, np.pi)
        
        self.weights = np.ones(self.num_particles) / self.num_particles

        self.publish_particles()

    def timer_callback(self):
        # TODO: Implementar maquina de estados para cada caso.
        # Debe haber estado para PLANNING, NAVIGATING y AVOIDING_OBSTACLE, pero pueden haber más estados si se desea.
        if not self.map_received:
            return

        # --- Localization (always running) ---
        if self.latest_scan is None:
            return
            
        if not self.initial_pose_received:
            self.initialize_particles_randomly()
            self.initial_pose_received = True
            return

        current_odom_tf = self.get_odom_transform()
        if current_odom_tf is None:
            if self.state in [State.NAVIGATING, State.AVOIDING_OBSTACLE]:
                self.stop_robot()
            return

        # Update particles with motion model if we have previous odometry
        if self.last_odom_pose is not None:
            self.motion_model(current_odom_tf)
        
        # Update particle weights with measurement model
        self.measurement_model()
        
        # Resample particles if needed
        self.resample()
        
        # Estimate current pose
        estimated_pose = self.estimate_pose()
        
        # State machine implementation
        if self.state == State.IDLE:
            # Robot is idle, waiting for a goal
            pass
            
        elif self.state == State.PLANNING:
            # Plan path to goal
            if self.goal_pose is not None:
                # TODO: Implement path planning here
                # For now, just transition to NAVIGATING
                self.get_logger().info("Planning complete. State -> NAVIGATING")
                self.state = State.NAVIGATING
                
        elif self.state == State.NAVIGATING:
            # Navigate along planned path
            if self.goal_pose is not None:
                # Check if goal is reached
                goal_distance = np.sqrt(
                    (estimated_pose.position.x - self.goal_pose.position.x)**2 + 
                    (estimated_pose.position.y - self.goal_pose.position.y)**2
                )
                
                if goal_distance < self.goal_tolerance:
                    self.get_logger().info("Goal reached! State -> IDLE")
                    self.state = State.IDLE
                    self.goal_pose = None
                    self.stop_robot()
                else:
                    # Check for obstacles
                    if self.detect_obstacle():
                        self.get_logger().info("Obstacle detected! State -> AVOIDING_OBSTACLE")
                        self.state = State.AVOIDING_OBSTACLE
                        self.obstacle_avoidance_active = True
                    else:
                        # Continue navigation
                        self.navigate_to_goal(estimated_pose)
                        
        elif self.state == State.AVOIDING_OBSTACLE:
            # Avoid obstacle
            if not self.detect_obstacle():
                self.get_logger().info("Obstacle cleared. State -> NAVIGATING")
                self.state = State.NAVIGATING
                self.obstacle_avoidance_active = False
            else:
                self.avoid_obstacle()

        # TODO: Implementar codigo para publicar la pose estimada, las particulas, y la transformacion entre el mapa y la base del robot.
        self.publish_pose(estimated_pose)
        self.publish_particles()
        self.publish_transform(estimated_pose, current_odom_tf)

    # ========== HELPER FUNCTIONS PARA TIMER CALLBACK ============
    def stop_robot(self):
        """Stop the robot by publishing zero velocities"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def detect_obstacle(self):
        """Detect obstacles using laser scan data"""
        if self.latest_scan is None:
            return False
        
        # Check front-facing laser readings for obstacles
        front_ranges = []
        total_readings = len(self.latest_scan.ranges)
        front_arc = int(total_readings * 0.2)  # 20% of readings in front
        
        # Get front readings (center ± front_arc/2)
        center = total_readings // 2
        start = center - front_arc // 2
        end = center + front_arc // 2
        
        for i in range(start, end):
            if i >= 0 and i < total_readings:
                range_val = self.latest_scan.ranges[i]
                if not np.isinf(range_val) and not np.isnan(range_val):
                    front_ranges.append(range_val)
        
        if front_ranges:
            min_distance = min(front_ranges)
            return min_distance < self.obstacle_detection_distance
        
        return False

    def navigate_to_goal(self, current_pose):
        """Navigate towards the goal using simple goal-seeking behavior"""
        if self.goal_pose is None:
            return
        
        # Calculate distance and angle to goal
        dx = self.goal_pose.position.x - current_pose.position.x
        dy = self.goal_pose.position.y - current_pose.position.y
        
        # Get current yaw
        q = current_pose.orientation
        current_yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        
        # Calculate desired yaw
        desired_yaw = np.arctan2(dy, dx)
        yaw_error = desired_yaw - current_yaw
        
        # Normalize angle
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi
        
        # Create velocity command
        twist = Twist()
        
        # Angular velocity (proportional control)
        twist.angular.z = 2.0 * yaw_error
        
        # Linear velocity (only move forward if roughly facing the goal)
        if abs(yaw_error) < 0.5:  # Within 30 degrees
            twist.linear.x = self.linear_velocity
        else:
            twist.linear.x = 0.0
        
        self.cmd_vel_pub.publish(twist)

    def avoid_obstacle(self):
        """Simple obstacle avoidance by turning"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = self.obstacle_avoidance_turn_speed
        self.cmd_vel_pub.publish(twist)
    
    #  ========== FIN DE LAS HELPER FUNCTIONS PARA TIMER CALLBACK ===========

    def get_odom_transform(self):
        try:
            return self.tf_buffer.lookup_transform(self.odom_frame_id, self.base_frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f'Could not get transform from {self.odom_frame_id} to {self.base_frame_id}. Skipping update. Error: {e}', throttle_duration_sec=2.0)
            return None

    def motion_model(self, current_odom_tf):
        current_odom_pose = current_odom_tf.transform
        
        # TODO: Implementar el modelo de movimiento para actualizar las particulas.
        
        # Extract current position and orientation
        curr_x = current_odom_pose.translation.x
        curr_y = current_odom_pose.translation.y
        curr_q = current_odom_pose.rotation
        curr_yaw = R.from_quat([curr_q.x, curr_q.y, curr_q.z, curr_q.w]).as_euler('xyz')[2]
        
        # Extract previous position and orientation
        prev_x = self.last_odom_pose.translation.x
        prev_y = self.last_odom_pose.translation.y
        prev_q = self.last_odom_pose.rotation
        prev_yaw = R.from_quat([prev_q.x, prev_q.y, prev_q.z, prev_q.w]).as_euler('xyz')[2]
        
        # Calculate odometry motion
        delta_x = curr_x - prev_x
        delta_y = curr_y - prev_y
        delta_yaw = curr_yaw - prev_yaw
        
        # Normalize angle difference
        while delta_yaw > np.pi:
            delta_yaw -= 2 * np.pi
        while delta_yaw < -np.pi:
            delta_yaw += 2 * np.pi
            
        # Calculate motion in robot's local frame
        delta_trans = np.sqrt(delta_x**2 + delta_y**2)
        delta_rot1 = np.arctan2(delta_y, delta_x) - prev_yaw
        delta_rot2 = delta_yaw - delta_rot1
        
        # Normalize angles
        while delta_rot1 > np.pi:
            delta_rot1 -= 2 * np.pi
        while delta_rot1 < -np.pi:
            delta_rot1 += 2 * np.pi
        while delta_rot2 > np.pi:
            delta_rot2 -= 2 * np.pi
        while delta_rot2 < -np.pi:
            delta_rot2 += 2 * np.pi
        
        # Apply motion model to each particle with noise
        for i in range(self.num_particles):
            # Add noise to motion commands based on alpha parameters
            noisy_rot1 = delta_rot1 + np.random.normal(0, self.alphas[0] * abs(delta_rot1) + self.alphas[1] * delta_trans)
            noisy_trans = delta_trans + np.random.normal(0, self.alphas[2] * delta_trans + self.alphas[3] * (abs(delta_rot1) + abs(delta_rot2)))
            noisy_rot2 = delta_rot2 + np.random.normal(0, self.alphas[0] * abs(delta_rot2) + self.alphas[1] * delta_trans)
            
            # Update particle position
            self.particles[i, 0] += noisy_trans * np.cos(self.particles[i, 2] + noisy_rot1)
            self.particles[i, 1] += noisy_trans * np.sin(self.particles[i, 2] + noisy_rot1)
            self.particles[i, 2] += noisy_rot1 + noisy_rot2
            
            # Normalize particle angle
            while self.particles[i, 2] > np.pi:
                self.particles[i, 2] -= 2 * np.pi
            while self.particles[i, 2] < -np.pi:
                self.particles[i, 2] += 2 * np.pi
        
        self.last_odom_pose = current_odom_pose

    def measurement_model(self):
        map_res = self.map_data.info.resolution
        map_origin = self.map_data.info.origin.position
        map_w = self.map_data.info.width
        map_h = self.map_data.info.height
        map_img = np.array(self.map_data.data).reshape((map_h, map_w))

        # Get laser scan parameters
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        ranges = np.array(self.latest_scan.ranges)
        
        # Skip invalid ranges and subsample for performance
        valid_indices = []
        for i, r in enumerate(ranges):
            if not np.isinf(r) and not np.isnan(r) and r > 0 and r < self.laser_max_range:
                valid_indices.append(i)
        
        # Subsample laser readings for computational efficiency (every 5th reading)
        valid_indices = valid_indices[::5]
        
        if len(valid_indices) == 0:
            return  # No valid laser readings
        
        # TODO: Implementar el modelo de medición para actualizar los pesos de las particulas por particula
        
        # Update weight for each particle
        for p_idx in range(self.num_particles):
            particle_x = self.particles[p_idx, 0]
            particle_y = self.particles[p_idx, 1]
            particle_theta = self.particles[p_idx, 2]
            
            log_likelihood = 0.0
            
            # Process each laser beam
            for beam_idx in valid_indices:
                # Calculate beam angle in world frame
                beam_angle = angle_min + beam_idx * angle_increment + particle_theta
                
                # Actual measured distance
                z_measured = ranges[beam_idx]
                
                # Calculate expected distance by ray casting in the map
                z_expected = self.ray_cast(particle_x, particle_y, beam_angle, map_img, map_res, map_origin, map_w, map_h)
                
                # Calculate likelihood using beam model
                if z_expected is not None:
                    # Hit probability (Gaussian around expected distance)
                    diff = z_measured - z_expected
                    p_hit = np.exp(-0.5 * (diff / 0.2)**2)  # sigma = 0.2m
                    
                    # Random probability (uniform)
                    p_rand = 1.0 / self.laser_max_range
                    
                    # Combined probability
                    p_total = self.z_hit * p_hit + self.z_rand * p_rand
                    
                    # Add log likelihood (avoid log(0))
                    if p_total > 1e-8:
                        log_likelihood += np.log(p_total)
                    else:
                        log_likelihood += np.log(1e-8)
            
            # Update particle weight
            self.weights[p_idx] = np.exp(log_likelihood)
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # If all weights are zero, reset to uniform
            self.weights = np.ones(self.num_particles) / self.num_particles
    # ============ HELPER FUNCTION PARA EL MEASURMENT MODEL =============
    def ray_cast(self, start_x, start_y, angle, map_img, resolution, origin, map_w, map_h):
        """Cast a ray from start position at given angle until hitting obstacle or max range"""
        # Ray casting parameters
        step_size = resolution * 0.5  # Half resolution for accuracy
        max_steps = int(self.laser_max_range / step_size)
        
        # Ray direction
        dx = np.cos(angle) * step_size
        dy = np.sin(angle) * step_size
        
        # Current position
        curr_x = start_x
        curr_y = start_y
        
        for step in range(max_steps):
            # Convert world coordinates to grid coordinates
            grid_x = int((curr_x - origin.x) / resolution)
            grid_y = int((curr_y - origin.y) / resolution)
            
            # Check bounds
            if grid_x < 0 or grid_x >= map_w or grid_y < 0 or grid_y >= map_h:
                break
            
            # Check if hit obstacle (occupied cell > 50)
            if map_img[grid_y, grid_x] > 50:
                # Calculate distance from start to hit point
                distance = np.sqrt((curr_x - start_x)**2 + (curr_y - start_y)**2)
                return distance
            
            # Move along ray
            curr_x += dx
            curr_y += dy
        
        # No obstacle hit within max range
        return self.laser_max_range

    def resample(self):
        # TODO: Implementar el resampleo de las particulas basado en los pesos.
        eff_sample_size = 1.0 / np.sum(self.weights**2)
        
        # Only resample if effective sample size is too low
        if eff_sample_size < self.num_particles / 2.0:
            # Low variance resampling
            new_particles = np.zeros_like(self.particles)
            cumulative_weights = np.cumsum(self.weights)
            
            # Generate random offset
            r = np.random.uniform(0, 1.0 / self.num_particles)
            
            i = 0
            for m in range(self.num_particles):
                u = r + m / self.num_particles
                while u > cumulative_weights[i]:
                    i += 1
                new_particles[m] = self.particles[i].copy()
                
            self.particles = new_particles
            self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_pose(self):
        # TODO: Implementar la estimación de pose a partir de las particulas y sus pesos.
        weighted_x = np.sum(self.particles[:, 0] * self.weights)
        weighted_y = np.sum(self.particles[:, 1] * self.weights)
        
        # For angle, use circular mean
        cos_sum = np.sum(np.cos(self.particles[:, 2]) * self.weights)
        sin_sum = np.sum(np.sin(self.particles[:, 2]) * self.weights)
        weighted_yaw = np.arctan2(sin_sum, cos_sum)
        
        # Create pose message
        pose = Pose()
        pose.position.x = weighted_x
        pose.position.y = weighted_y
        pose.position.z = 0.0
        
        # Convert yaw to quaternion
        q = R.from_euler('z', weighted_yaw).as_quat()
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        return pose

    def publish_pose(self, estimated_pose):
        p = PoseWithCovarianceStamped()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = self.map_frame_id
        p.pose.pose = estimated_pose
        self.pose_pub.publish(p)

    def publish_particles(self):
        ma = MarkerArray()
        for i, p in enumerate(self.particles):
            marker = Marker()
            marker.header.frame_id = self.map_frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "particles"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = p[0]
            marker.pose.position.y = p[1]
            q = R.from_euler('z', p[2]).as_quat()
            marker.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            marker.scale.x = 0.1
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.a = 0.5
            marker.color.r = 1.0
            ma.markers.append(marker)
        self.particle_pub.publish(ma)

    def publish_transform(self, estimated_pose, odom_tf):
        map_to_base_mat = self.pose_to_matrix(estimated_pose)
        odom_to_base_mat = self.transform_to_matrix(odom_tf.transform)
        map_to_odom_mat = np.dot(map_to_base_mat, np.linalg.inv(odom_to_base_mat))
        
        t = TransformStamped()
        
        # TODO: Completar el TransformStamped con la transformacion entre el mapa y la base del robot.
        
        # Set header information
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame_id
        t.child_frame_id = self.odom_frame_id
        
        # Extract translation from transformation matrix
        t.transform.translation.x = map_to_odom_mat[0, 3]
        t.transform.translation.y = map_to_odom_mat[1, 3]
        t.transform.translation.z = map_to_odom_mat[2, 3]
        
        # Extract rotation from transformation matrix and convert to quaternion
        rotation_matrix = map_to_odom_mat[:3, :3]
        r = R.from_matrix(rotation_matrix)
        q = r.as_quat()  # Returns [x, y, z, w]
        
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def pose_to_matrix(self, pose):
        q = pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        mat = np.eye(4)
        mat[:3, :3] = r.as_matrix()
        mat[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return mat

    def transform_to_matrix(self, transform):
        q = transform.rotation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        mat = np.eye(4)
        mat[:3, :3] = r.as_matrix()
        t = transform.translation
        mat[:3, 3] = [t.x, t.y, t.z]
        return mat

    def world_to_grid(self, wx, wy):
        gx = int((wx - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        gy = int((wy - self.map_data.info.origin.position.y) / self.map_data.info.resolution)
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        wx = gx * self.map_data.info.resolution + self.map_data.info.origin.position.x
        wy = gy * self.map_data.info.resolution + self.map_data.info.origin.position.y
        return (wx, wy)
    

    def publish_path(self, path_msg):
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame_id
        self.path_pub.publish(path_msg)

    

def main(args=None):
    rclpy.init(args=args)
    node = AmclNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
