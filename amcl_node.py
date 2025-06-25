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
        self.declare_parameter('goal_tolerance', 0.1)
        self.declare_parameter('path_pruning_distance', 0.3)
        self.declare_parameter('safety_margin_cells', 3)

        
        
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
        
        # Extract position and orientation from initial pose
        x = initial_pose.position.x
        y = initial_pose.position.y
        q = initial_pose.orientation
        yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        
        # Initialize particles with Gaussian noise around the initial pose
        position_std = 0.1  # Standard deviation for position (meters)
        orientation_std = 0.1  # Standard deviation for orientation (radians)
        
        # Generate particles with noise
        self.particles[:, 0] = np.random.normal(x, position_std, self.num_particles)
        self.particles[:, 1] = np.random.normal(y, position_std, self.num_particles)
        self.particles[:, 2] = np.random.normal(yaw, orientation_std, self.num_particles)
        
        # Initialize uniform weights
        self.weights = np.ones(self.num_particles) / self.num_particles

        self.publish_particles()

    def initialize_particles_randomly(self):
        # TODO: Inizializar particulas aleatoriamente en todo el mapa
        
        # Get map bounds
        map_w = self.map_data.info.width
        map_h = self.map_data.info.height
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        
        # Find free space cells in the map
        free_cells = []
        for y in range(map_h):
            for x in range(map_w):
                if self.grid[y, x] <= -1.3:  # Free space
                    world_x = x * resolution + origin_x
                    world_y = y * resolution + origin_y
                    free_cells.append((world_x, world_y))
        
        if len(free_cells) == 0:
            self.get_logger().warn("No free space found in map for particle initialization")
            return
            
        # Randomly sample from free cells
        selected_indices = np.random.choice(len(free_cells), self.num_particles, replace=True)
        
        for i, idx in enumerate(selected_indices):
            self.particles[i, 0] = free_cells[idx][0]
            self.particles[i, 1] = free_cells[idx][1]
            self.particles[i, 2] = np.random.uniform(-np.pi, np.pi)
        
        # Initialize uniform weights
        self.weights = np.ones(self.num_particles) / self.num_particles

        self.publish_particles()

    def timer_callback(self):
        # TODO: Implementar maquina de estados para cada caso.
        # Debe haber estado para PLANNING, NAVIGATING y AVOIDING_OBSTACLE, pero pueden haber más estados si se desea.
        if not self.map_received:
            print("Map not received yet. Cannot proceed with localization.")
            return

        # --- Localization (always running) ---
        if self.latest_scan is None:
            print("No scan data received yet. Cannot proceed with localization.")
            return
            
        if not self.initial_pose_received:
            self.initialize_particles_randomly()
            self.initial_pose_received = True
            print("Initial pose not received yet. Initializing particles randomly.")
            return

        current_odom_tf = self.get_odom_transform()
        if current_odom_tf is None:
            if self.state in [State.NAVIGATING, State.AVOIDING_OBSTACLE]:
                # Stop robot by publishing zero velocity
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd)
            return

        # --- AMCL Localization Algorithm (Core Loop) ---
        
        # 1. Motion Model: Update particle positions based on robot movement
        if self.last_odom_pose is not None:
            self.motion_model(current_odom_tf)
        
        # 2. Measurement Model: Update particle weights based on laser scan
        self.measurement_model()
        
        # 3. Resample particles based on weights
        self.resample()
        
        # 4. Estimate robot pose from weighted particles
        estimated_pose = self.estimate_pose()

        # --- Simple State Machine for Navigation ---
        
        if self.state == State.IDLE:
            # Wait for goal or do nothing
            pass
            
        elif self.state == State.PLANNING:
            if self.goal_pose is not None:
                # Simple planning: just set state to navigating
                self.state = State.NAVIGATING
                self.get_logger().info("State -> NAVIGATING")
                    
        elif self.state == State.NAVIGATING:
            if self.goal_pose is not None:
                # Check if goal is reached
                distance_to_goal = np.sqrt((estimated_pose.position.x - self.goal_pose.position.x)**2 + 
                                         (estimated_pose.position.y - self.goal_pose.position.y)**2)
                
                if distance_to_goal < self.goal_tolerance:
                    # Stop robot
                    cmd = Twist()
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
                    self.cmd_vel_pub.publish(cmd)
                    self.state = State.IDLE
                    self.get_logger().info("Goal reached! State -> IDLE")
                    return
                    
                # Simple navigation: move towards goal
                dx = self.goal_pose.position.x - estimated_pose.position.x
                dy = self.goal_pose.position.y - estimated_pose.position.y
                
                # Current yaw
                current_q = estimated_pose.orientation
                current_yaw = R.from_quat([current_q.x, current_q.y, current_q.z, current_q.w]).as_euler('xyz')[2]
                
                # Angle to goal
                target_yaw = np.arctan2(dy, dx)
                angle_diff = target_yaw - current_yaw
                
                # Normalize angle
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                
                # Simple control
                cmd = Twist()
                cmd.linear.x = self.linear_velocity
                cmd.angular.z = 2.0 * angle_diff
                self.cmd_vel_pub.publish(cmd)
            else:
                self.state = State.IDLE
                
        elif self.state == State.AVOIDING_OBSTACLE:
            # Simple obstacle avoidance: just turn
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = self.obstacle_avoidance_turn_speed
            self.cmd_vel_pub.publish(cmd)
            
            # Switch back to navigating after some time (simplified)
            self.state = State.NAVIGATING

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
        
        if self.last_odom_pose is None:
            self.last_odom_pose = current_odom_pose
            return
            
        # Calculate motion difference in odometry frame
        last_x = self.last_odom_pose.translation.x
        last_y = self.last_odom_pose.translation.y
        last_q = self.last_odom_pose.rotation
        last_yaw = R.from_quat([last_q.x, last_q.y, last_q.z, last_q.w]).as_euler('xyz')[2]
        
        curr_x = current_odom_pose.translation.x
        curr_y = current_odom_pose.translation.y
        curr_q = current_odom_pose.rotation
        curr_yaw = R.from_quat([curr_q.x, curr_q.y, curr_q.z, curr_q.w]).as_euler('xyz')[2]
        
        # Calculate motion in odometry frame
        delta_x = curr_x - last_x
        delta_y = curr_y - last_y
        delta_yaw = curr_yaw - last_yaw
        
        # Normalize angle
        while delta_yaw > np.pi:
            delta_yaw -= 2 * np.pi
        while delta_yaw < -np.pi:
            delta_yaw += 2 * np.pi
            
        # Calculate motion parameters
        delta_trans = np.sqrt(delta_x**2 + delta_y**2)
        delta_rot1 = 0.0
        delta_rot2 = 0.0
        
        if delta_trans > 1e-6:
            delta_rot1 = np.arctan2(delta_y, delta_x) - last_yaw
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
            
        # Apply motion model to each particle
        for i in range(self.num_particles):
            # Add noise to motion parameters
            noise_rot1 = np.random.normal(0, self.alphas[0] * abs(delta_rot1) + self.alphas[1] * delta_trans)
            noise_trans = np.random.normal(0, self.alphas[2] * delta_trans + self.alphas[3] * (abs(delta_rot1) + abs(delta_rot2)))
            noise_rot2 = np.random.normal(0, self.alphas[0] * abs(delta_rot2) + self.alphas[1] * delta_trans)
            
            # Apply noisy motion
            rot1_hat = delta_rot1 + noise_rot1
            trans_hat = delta_trans + noise_trans
            rot2_hat = delta_rot2 + noise_rot2
            
            # Update particle pose
            self.particles[i, 0] += trans_hat * np.cos(self.particles[i, 2] + rot1_hat)
            self.particles[i, 1] += trans_hat * np.sin(self.particles[i, 2] + rot1_hat)
            self.particles[i, 2] += rot1_hat + rot2_hat
            
            # Normalize angle
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

        if self.latest_scan is None:
            return

        # TODO: Implementar el modelo de medición para actualizar los pesos de las particulas por particula
        
        scan_ranges = np.array(self.latest_scan.ranges)
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        
        # Filter out invalid readings
        valid_indices = np.where((scan_ranges > 0) & (scan_ranges < self.laser_max_range))[0]
        
        if len(valid_indices) == 0:
            return
            
        # Subsample scan for efficiency (use every nth beam)
        step = max(1, len(valid_indices) // 50)  # Use at most 50 beams
        selected_indices = valid_indices[::step]
        
        for i in range(self.num_particles):
            particle_x = self.particles[i, 0]
            particle_y = self.particles[i, 1]
            particle_yaw = self.particles[i, 2]
            
            weight = 1.0
            
            for beam_idx in selected_indices:
                # Calculate beam angle
                beam_angle = angle_min + beam_idx * angle_increment + particle_yaw
                
                # Expected range from particle pose
                expected_range = self.ray_cast(particle_x, particle_y, beam_angle, map_img, map_res, map_origin, map_w, map_h)
                
                if expected_range > 0:
                    # Measured range
                    measured_range = scan_ranges[beam_idx]
                    
                    # Calculate likelihood using beam model
                    likelihood = self.beam_model(measured_range, expected_range)
                    weight *= likelihood
                    
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
        
        # Weighted average of particles
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
        
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame_id
        t.child_frame_id = self.odom_frame_id
        
        # Extract translation and rotation from transformation matrix
        t.transform.translation.x = map_to_odom_mat[0, 3]
        t.transform.translation.y = map_to_odom_mat[1, 3]
        t.transform.translation.z = map_to_odom_mat[2, 3]
        
        # Convert rotation matrix to quaternion
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

    

def main(args=None):
    rclpy.init(args=args)
    node = AmclNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 