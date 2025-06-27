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
        self.declare_parameter('num_particles', 500)
        self.declare_parameter('alpha1', 0.2)
        self.declare_parameter('alpha2', 0.2)
        self.declare_parameter('alpha3', 0.2)
        self.declare_parameter('alpha4', 0.2)
        self.declare_parameter('z_hit', 0.8)
        self.declare_parameter('z_rand', 0.2)
        self.declare_parameter('lookahead_distance', 0.5)
        self.declare_parameter('linear_velocity', 0.2)
        self.declare_parameter('goal_tolerance', 0.2)
        self.declare_parameter('path_pruning_distance', 0.3)
        self.declare_parameter('safety_margin_cells', 2)

        
        
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
        
        # Extract initial position and orientation
        x = initial_pose.position.x
        y = initial_pose.position.y
        q = initial_pose.orientation
        initial_yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        
        # Initialize particles around the initial pose with Gaussian noise
        std_xy = 0.5  # Standard deviation for position (meters)
        std_yaw = 0.3  # Standard deviation for orientation (radians)
        
        self.particles[:, 0] = np.random.normal(x, std_xy, self.num_particles)
        self.particles[:, 1] = np.random.normal(y, std_xy, self.num_particles)
        self.particles[:, 2] = np.random.normal(initial_yaw, std_yaw, self.num_particles)
        
        # Normalize angles to [-pi, pi]
        self.particles[:, 2] = np.arctan2(np.sin(self.particles[:, 2]), np.cos(self.particles[:, 2]))
        
        # Initialize uniform weights
        self.weights = np.ones(self.num_particles) / self.num_particles

        self.publish_particles()

    def initialize_particles_randomly(self):
        # TODO: Inizializar particulas aleatoriamente en todo el mapa
        
        # Get free space positions from the map
        free_cells = np.where(self.grid == 0)  # Free space cells
        if len(free_cells[0]) == 0:
            self.get_logger().warn("No free space found in map, using default initialization")
            # Fallback to center of map
            center_x = self.map_data.info.width / 2
            center_y = self.map_data.info.height / 2
            self.particles[:, 0] = center_x * self.map_data.info.resolution + self.map_data.info.origin.position.x
            self.particles[:, 1] = center_y * self.map_data.info.resolution + self.map_data.info.origin.position.y
        else:
            # Randomly sample from free cells
            num_free = len(free_cells[0])
            sample_indices = np.random.choice(num_free, self.num_particles, replace=True)
            
            # Convert grid coordinates to world coordinates
            for i in range(self.num_particles):
                gx = free_cells[1][sample_indices[i]]  # x is columns (width)
                gy = free_cells[0][sample_indices[i]]  # y is rows (height)
                wx, wy = self.grid_to_world(gx, gy)
                self.particles[i, 0] = wx
                self.particles[i, 1] = wy
        
        # Random orientations
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)
        
        # Initialize uniform weights
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

        # Update particle filter
        if self.last_odom_pose is not None:
            self.motion_model(current_odom_tf)
        self.measurement_model()
        self.resample()
        estimated_pose = self.estimate_pose()

        # State machine for navigation
        if self.state == State.PLANNING:
            if self.goal_pose is not None:
                path = self.a_star_planning(estimated_pose, self.goal_pose)
                if path is not None:
                    self.current_path = path
                    self.state = State.NAVIGATING
                    self.get_logger().info("Path planned successfully. State -> NAVIGATING")
                    # Publish path for visualization
                    path_msg = self.create_path_msg(path)
                    self.publish_path(path_msg)
                else:
                    self.get_logger().error("Failed to plan path. State -> IDLE")
                    self.state = State.IDLE
                    self.stop_robot()

        elif self.state == State.NAVIGATING:
            if self.current_path is None:
                self.state = State.IDLE
                self.stop_robot()
                return
            if self.detect_obstacle():
                self.get_logger().warn("Obstacle detected. State -> AVOIDING_OBSTACLE")
                self.state = State.AVOIDING_OBSTACLE
                self.obstacle_avoidance_active = True
                q = estimated_pose.orientation
                self.obstacle_avoidance_start_yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
                self.obstacle_avoidance_last_yaw = None  # Add this line
                self.obstacle_avoidance_cumulative_angle = 0.0
                return

            # Check if goal is reached
            goal_distance = self.distance_to_goal(estimated_pose, self.goal_pose)
            if goal_distance < self.goal_tolerance:
                self.get_logger().info("Goal reached! State -> IDLE")
                self.state = State.IDLE
                self.stop_robot()
                return

            # Check for obstacles
            if self.detect_obstacle():
                self.get_logger().warn("Obstacle detected. State -> AVOIDING_OBSTACLE")
                self.state = State.AVOIDING_OBSTACLE
                self.obstacle_avoidance_active = True
                q = estimated_pose.orientation
                self.obstacle_avoidance_start_yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
                self.obstacle_avoidance_cumulative_angle = 0.0
                return

            # Follow path using pure pursuit
            cmd = self.pure_pursuit_control(estimated_pose, self.current_path)
            self.cmd_vel_pub.publish(cmd)

        elif self.state == State.AVOIDING_OBSTACLE:
            if not self.detect_obstacle():
                # Clear path, return to navigation
                self.get_logger().info("Obstacle cleared. State -> NAVIGATING")
                self.state = State.NAVIGATING
                self.obstacle_avoidance_active = False
                return

            # Simple obstacle avoidance: turn left
            cmd = Twist()
            cmd.linear.x = 0.05  # Move slowly forward
            cmd.angular.z = self.obstacle_avoidance_turn_speed
            self.cmd_vel_pub.publish(cmd)

            # Track rotation to avoid infinite spinning
            q = estimated_pose.orientation
            current_yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
            
            if self.obstacle_avoidance_last_yaw is not None:
                angle_diff = self.normalize_angle(current_yaw - self.obstacle_avoidance_last_yaw)
                self.obstacle_avoidance_cumulative_angle += abs(angle_diff)
            
            self.obstacle_avoidance_last_yaw = current_yaw
            
            # If we've turned too much, try replanning
            if self.obstacle_avoidance_cumulative_angle > 2 * np.pi:
                self.get_logger().warn("Obstacle avoidance timeout. State -> PLANNING")
                self.state = State.PLANNING
                self.obstacle_avoidance_active = False

        # TODO: Implementar codigo para publicar la pose estimada, las particulas, y la transformacion entre el mapa y la base del robot.

        self.publish_pose(estimated_pose)
        self.publish_particles()
        self.publish_transform(estimated_pose, current_odom_tf)


    def get_odom_transform(self):
        try:
            return self.tf_buffer.lookup_transform(self.odom_frame_id, self.base_frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f'Could not get transform from {self.odom_frame_id} to {self.base_frame_id}. Skipping update. Error: {e}', throttle_duration_sec=2.0)
            return None

    def motion_model(self, current_odom_tf):
        current_odom_pose = current_odom_tf.transform
        
        # TODO: Implementar el modelo de movimiento para actualizar las particulas.
        
        # Calculate odometry change
        if self.last_odom_pose is None:
            self.last_odom_pose = current_odom_pose
            return
        
        # Extract positions and orientations
        prev_x = self.last_odom_pose.translation.x
        prev_y = self.last_odom_pose.translation.y
        prev_q = self.last_odom_pose.rotation
        prev_yaw = R.from_quat([prev_q.x, prev_q.y, prev_q.z, prev_q.w]).as_euler('xyz')[2]
        
        curr_x = current_odom_pose.translation.x
        curr_y = current_odom_pose.translation.y
        curr_q = current_odom_pose.rotation
        curr_yaw = R.from_quat([curr_q.x, curr_q.y, curr_q.z, curr_q.w]).as_euler('xyz')[2]
        
        # Calculate odometry deltas
        delta_x = curr_x - prev_x
        delta_y = curr_y - prev_y
        delta_yaw = self.normalize_angle(curr_yaw - prev_yaw)
        
        # Calculate motion parameters
        delta_trans = np.sqrt(delta_x**2 + delta_y**2)
        delta_rot1 = self.normalize_angle(np.arctan2(delta_y, delta_x) - prev_yaw)
        delta_rot2 = self.normalize_angle(delta_yaw - delta_rot1)
        
        # Skip if no significant motion
        if delta_trans < 0.001 and abs(delta_yaw) < 0.001:
            self.last_odom_pose = current_odom_pose
            return
        
        # Apply motion model to each particle
        for i in range(self.num_particles):
            # Add noise to motion parameters
            noisy_rot1 = delta_rot1 + np.random.normal(0, self.alphas[0] * abs(delta_rot1) + self.alphas[1] * delta_trans)
            noisy_trans = delta_trans + np.random.normal(0, self.alphas[2] * delta_trans + self.alphas[3] * (abs(delta_rot1) + abs(delta_rot2)))
            noisy_rot2 = delta_rot2 + np.random.normal(0, self.alphas[0] * abs(delta_rot2) + self.alphas[1] * delta_trans)
            
            # Update particle pose
            self.particles[i, 0] += noisy_trans * np.cos(self.particles[i, 2] + noisy_rot1)
            self.particles[i, 1] += noisy_trans * np.sin(self.particles[i, 2] + noisy_rot1)
            self.particles[i, 2] = self.normalize_angle(self.particles[i, 2] + noisy_rot1 + noisy_rot2)
        
        self.last_odom_pose = current_odom_pose

    def measurement_model(self):
        map_res = self.map_data.info.resolution
        map_origin = self.map_data.info.origin.position
        map_w = self.map_data.info.width
        map_h = self.map_data.info.height
        map_img = np.array(self.map_data.data).reshape((map_h, map_w))

        if self.latest_scan is None:
            return

        # Get laser scan parameters
        ranges = np.array(self.latest_scan.ranges)
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        range_max = min(self.latest_scan.range_max, self.laser_max_range)
        
        # Filter valid measurements
        valid_indices = np.where((ranges > 0) & (ranges < range_max) & np.isfinite(ranges))[0]
        
        if len(valid_indices) == 0:
            return

        # TODO: Implementar el modelo de medición para actualizar los pesos de las particulas por particula
        
        # Process each particle
        for i in range(self.num_particles):
            particle_x = self.particles[i, 0]
            particle_y = self.particles[i, 1]
            particle_yaw = self.particles[i, 2]
            
            weight = 1.0
            
            # Sample a subset of rays for efficiency
            step = max(1, len(valid_indices) // 20)  # Use max 20 rays
            
            for j in valid_indices[::step]:
                measured_range = ranges[j]
                ray_angle = angle_min + j * angle_increment + particle_yaw
                
                # Ray tracing to find expected range
                expected_range = self.ray_trace(particle_x, particle_y, ray_angle, range_max)
                
                if expected_range > 0:
                    # Likelihood based on range difference
                    range_diff = abs(measured_range - expected_range)
                    
                    # Simple likelihood model
                    if range_diff < 0.1:  # Hit
                        weight *= self.z_hit
                    else:  # Miss or random
                        weight *= self.z_rand / range_max
            
            self.weights[i] = weight
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

    def resample(self):
        # TODO: Implementar el resampleo de las particulas basado en los pesos.
        
        # Calculate effective sample size
        eff_sample_size = 1.0 / np.sum(self.weights**2)
        
        # Only resample if effective sample size is too low
        if eff_sample_size < self.num_particles / 2:
            # Low variance resampling
            new_particles = np.zeros_like(self.particles)
            
            # Generate cumulative sum
            cumsum = np.cumsum(self.weights)
            
            # Generate random starting point
            r = np.random.uniform(0, 1.0/self.num_particles)
            
            i = 0
            for j in range(self.num_particles):
                u = r + j / self.num_particles
                
                # Find particle to resample
                while u > cumsum[i]:
                    i += 1
                
                new_particles[j] = self.particles[i].copy()
            
            self.particles = new_particles
            self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_pose(self):
        # TODO: Implementar la estimación de pose a partir de las particulas y sus pesos.
        
        # Weighted average of particles
        if np.sum(self.weights) == 0:
            # If all weights are zero, use uniform weights
            weights = np.ones(self.num_particles) / self.num_particles
        else:
            weights = self.weights / np.sum(self.weights)
        
        # Calculate weighted position
        x = np.sum(weights * self.particles[:, 0])
        y = np.sum(weights * self.particles[:, 1])
        
        # Calculate weighted orientation (using circular statistics)
        cos_sum = np.sum(weights * np.cos(self.particles[:, 2]))
        sin_sum = np.sum(weights * np.sin(self.particles[:, 2]))
        yaw = np.arctan2(sin_sum, cos_sum)
        
        # Create pose message
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        
        # Convert yaw to quaternion
        q = R.from_euler('z', yaw).as_quat()
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
        
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame_id
        t.child_frame_id = self.odom_frame_id
        
        # Extract translation
        t.transform.translation.x = map_to_odom_mat[0, 3]
        t.transform.translation.y = map_to_odom_mat[1, 3]
        t.transform.translation.z = map_to_odom_mat[2, 3]
        
        # Extract rotation
        rotation_matrix = map_to_odom_mat[:3, :3]
        r = R.from_matrix(rotation_matrix)
        q = r.as_quat()
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

    def inflate_map(self):
        """Inflate the map for safe navigation"""
        if self.grid is None:
            return
        
        from scipy.ndimage import binary_dilation, distance_transform_edt
        
        # Convert occupancy grid to binary (obstacle/free)
        obstacle_mask = self.grid > 50  # Cells with >50% probability are obstacles
        unknown_mask = self.grid == -1  # Unknown cells
        
        # Treat unknown as obstacles for safety
        obstacle_or_unknown = obstacle_mask | unknown_mask
        
        # Create structuring element for inflation
        struct_size = self.safety_margin_cells * 2 + 1
        struct = np.ones((struct_size, struct_size))
        
        # Inflate obstacles
        inflated_mask = binary_dilation(obstacle_or_unknown, structure=struct)
        
        # Convert back to occupancy grid format
        self.inflated_grid = np.zeros_like(self.grid)
        self.inflated_grid[inflated_mask] = 100
        self.inflated_grid[self.grid == -1] = -1  # Keep unknown cells marked

    def a_star_planning(self, start_pose, goal_pose):
        """A* path planning algorithm"""
        start_gx, start_gy = self.world_to_grid(start_pose.position.x, start_pose.position.y)
        goal_gx, goal_gy = self.world_to_grid(goal_pose.position.x, goal_pose.position.y)
        
        # Check if start and goal are valid
        if not self.is_valid_cell(start_gx, start_gy) or not self.is_valid_cell(goal_gx, goal_gy):
            self.get_logger().error("Start or goal position is invalid")
            return None
        
        # A* implementation
        open_set = [(0, start_gx, start_gy)]
        came_from = {}
        g_score = {(start_gx, start_gy): 0}
        f_score = {(start_gx, start_gy): self.heuristic(start_gx, start_gy, goal_gx, goal_gy)}
        
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        while open_set:
            current_f, current_x, current_y = heapq.heappop(open_set)
            
            if current_x == goal_gx and current_y == goal_gy:
                # Reconstruct path
                path = []
                while (current_x, current_y) in came_from:
                    wx, wy = self.grid_to_world(current_x, current_y)
                    path.append((wx, wy))
                    current_x, current_y = came_from[(current_x, current_y)]
                
                # Add start position
                wx, wy = self.grid_to_world(start_gx, start_gy)
                path.append((wx, wy))
                path.reverse()
                
                return self.smooth_path(path)
            
            for dx, dy in directions:
                neighbor_x, neighbor_y = current_x + dx, current_y + dy
                
                if not self.is_valid_cell(neighbor_x, neighbor_y):
                    continue
                
                # Calculate cost (diagonal moves cost more)
                move_cost = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
                tentative_g = g_score[(current_x, current_y)] + move_cost
                
                if (neighbor_x, neighbor_y) not in g_score or tentative_g < g_score[(neighbor_x, neighbor_y)]:
                    came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                    g_score[(neighbor_x, neighbor_y)] = tentative_g
                    f_score[(neighbor_x, neighbor_y)] = tentative_g + self.heuristic(neighbor_x, neighbor_y, goal_gx, goal_gy)
                    heapq.heappush(open_set, (f_score[(neighbor_x, neighbor_y)], neighbor_x, neighbor_y))
        
        return None  # No path found

    def heuristic(self, x1, y1, x2, y2):
        """Euclidean distance heuristic for A*"""
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def is_valid_cell(self, gx, gy):
        """Check if grid cell is valid and free"""
        if gx < 0 or gx >= self.map_data.info.width or gy < 0 or gy >= self.map_data.info.height:
            return False
        
        # Use inflated grid for planning
        if self.inflated_grid[gy, gx] > 50:  # Obstacle or inflated area
            return False
        
        return True

    def smooth_path(self, path):
        """Simple path smoothing"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        
        for i in range(1, len(path) - 1):
            # Check if we can skip this waypoint
            if not self.line_of_sight(smoothed[-1], path[i + 1]):
                smoothed.append(path[i])
        
        smoothed.append(path[-1])
        return smoothed

    def line_of_sight(self, start, end):
        """Check if there's a clear line of sight between two points"""
        x0, y0 = self.world_to_grid(start[0], start[1])
        x1, y1 = self.world_to_grid(end[0], end[1])
        
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            if not self.is_valid_cell(x, y):
                return False
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True

    def pure_pursuit_control(self, current_pose, path):
        """Pure pursuit controller for path following"""
        if not path or len(path) < 2:
            return Twist()
        
        # Prune path points that are behind the robot
        path = self.prune_path(current_pose, path)
        if not path:
            return Twist()
        
        # Find lookahead point
        lookahead_point = self.find_lookahead_point(current_pose, path)
        if lookahead_point is None:
            return Twist()
        
        # Calculate control commands
        cmd = Twist()
        
        # Calculate distance and angle to lookahead point
        dx = lookahead_point[0] - current_pose.position.x
        dy = lookahead_point[1] - current_pose.position.y
        distance_to_lookahead = np.sqrt(dx*dx + dy*dy)
        
        # Get current yaw
        q = current_pose.orientation
        current_yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        
        # Calculate desired heading
        desired_yaw = np.arctan2(dy, dx)
        yaw_error = self.normalize_angle(desired_yaw - current_yaw)
        
        # Adaptive linear velocity based on yaw error
        if abs(yaw_error) > np.pi/4:  # If error > 45 degrees
            cmd.linear.x = 0.05  # Move slowly
        else:
            # Scale linear velocity based on yaw error
            cmd.linear.x = self.linear_velocity * (1.0 - 0.5 * abs(yaw_error) / (np.pi/4))
        
        # Proportional-Derivative control for angular velocity
        kp = 2.0  # Proportional gain
        cmd.angular.z = kp * yaw_error
        
        # Limit angular velocity
        max_angular = 1.0
        cmd.angular.z = np.clip(cmd.angular.z, -max_angular, max_angular)
        
        return cmd

    def prune_path(self, current_pose, path):
        """Remove path points that are behind the robot or too close"""
        if not path:
            return path
        
        current_x = current_pose.position.x
        current_y = current_pose.position.y
        
        # Find the closest point ahead of the robot
        pruned_path = []
        for i, (px, py) in enumerate(path):
            dist = np.sqrt((px - current_x)**2 + (py - current_y)**2)
            
            # Keep points that are ahead and not too close
            if dist > self.path_pruning_distance:
                pruned_path = path[i:]
                break
        
        return pruned_path if pruned_path else [path[-1]]  # Always keep the goal

    def find_lookahead_point(self, current_pose, path):
        """Find the lookahead point on the path"""
        if not path:
            return None
        
        current_x = current_pose.position.x
        current_y = current_pose.position.y
        
        # Find the closest point on the path
        min_dist = float('inf')
        closest_idx = 0
        
        for i, (px, py) in enumerate(path):
            dist = np.sqrt((px - current_x)**2 + (py - current_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Find lookahead point
        for i in range(closest_idx, len(path)):
            px, py = path[i]
            dist = np.sqrt((px - current_x)**2 + (py - current_y)**2)
            
            if dist >= self.lookahead_distance:
                return (px, py)
        
        # If no point found, return the last point
        return path[-1]

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

    def distance_to_goal(self, current_pose, goal_pose):
        """Calculate distance to goal"""
        dx = goal_pose.position.x - current_pose.position.x
        dy = goal_pose.position.y - current_pose.position.y
        return np.sqrt(dx*dx + dy*dy)

    def detect_obstacle(self):
        """Detect obstacles using laser scan"""
        if self.latest_scan is None:
            return False
        
        ranges = np.array(self.latest_scan.ranges)
        
        # Calculate the indices for front 60 degrees (-30 to +30)
        num_rays = len(ranges)
        angle_min = self.latest_scan.angle_min
        angle_max = self.latest_scan.angle_max
        angle_increment = self.latest_scan.angle_increment
        
        # Find indices for -30 and +30 degrees
        front_angle_min = -np.pi/6  # -30 degrees
        front_angle_max = np.pi/6   # +30 degrees
        
        front_start_idx = int((front_angle_min - angle_min) / angle_increment)
        front_end_idx = int((front_angle_max - angle_min) / angle_increment)
        
        # Ensure indices are within bounds
        front_start_idx = max(0, front_start_idx)
        front_end_idx = min(num_rays - 1, front_end_idx)
        
        # Get front ranges
        front_ranges = ranges[front_start_idx:front_end_idx+1]
        valid_front_ranges = front_ranges[np.isfinite(front_ranges) & (front_ranges > 0)]
        
        if len(valid_front_ranges) > 0:
            min_distance = np.min(valid_front_ranges)
            return min_distance < self.obstacle_detection_distance
        
        return False

    def create_path_msg(self, path):
        """Create a Path message from list of waypoints"""
        path_msg = Path()
        path_msg.header.frame_id = self.map_frame_id
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y in path:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = self.map_frame_id
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            path_msg.poses.append(pose_stamped)
        
        return path_msg

    def ray_trace(self, start_x, start_y, angle, max_range):
        """Ray tracing to find expected range measurement"""
        step = self.map_data.info.resolution / 2  # Sub-pixel accuracy
        
        for r in np.arange(0, max_range, step):
            x = start_x + r * np.cos(angle)
            y = start_y + r * np.sin(angle)
            
            gx, gy = self.world_to_grid(x, y)
            
            # Check bounds
            if gx < 0 or gx >= self.map_data.info.width or gy < 0 or gy >= self.map_data.info.height:
                return max_range
            
            # Check if hit obstacle
            if self.grid[gy, gx] > 50:  # Obstacle
                return r
        
        return max_range

def main(args=None):
    rclpy.init(args=args)
    node = AmclNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
