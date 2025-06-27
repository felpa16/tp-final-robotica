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
        
        self.last_odom_pose = current_odom_pose

    def measurement_model(self):
        map_res = self.map_data.info.resolution
        map_origin = self.map_data.info.origin.position
        map_w = self.map_data.info.width
        map_h = self.map_data.info.height
        map_img = np.array(self.map_data.data).reshape((map_h, map_w))





        # TODO: Implementar el modelo de medición para actualizar los pesos de las particulas por particula

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
