#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import tf_transformations
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class Particle:
    def __init__(self, x, y, theta, weight, map_shape):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
        self.log_odds_map = np.zeros(map_shape, dtype=np.float32)

    def pose(self):
        return np.array([self.x, self.y, self.theta])

class PythonSlamNode(Node):
    def __init__(self):
        super().__init__('python_slam_node')

        # Parameters
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        # TODO: define map resolution, width, height, and number of particles

        self.declare_parameter('map_resolution', 0.05)  # meters per cell
        self.declare_parameter('map_width_meters', 10.0)
        self.declare_parameter('map_height_meters', 10.0)
        self.declare_parameter('num_particles', 100)


        self.resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.map_width_m = self.get_parameter('map_width_meters').get_parameter_value().double_value
        self.map_height_m = self.get_parameter('map_height_meters').get_parameter_value().double_value
        self.map_width_cells = int(self.map_width_m / self.resolution)
        self.map_height_cells = int(self.map_height_m / self.resolution)
        self.map_origin_x = -self.map_width_m / 2.0
        self.map_origin_y = -5.0

        # TODO: define the log-odds criteria for free and occupied cells
        log_odds_occupied = 0.8
        log_odds_free = 0.2
        self.log_odds_free = np.log(log_odds_free / (1.0 - log_odds_free))  # -1.39
        self.log_odds_occupied = np.log(log_odds_occupied / (1.0 - log_odds_occupied)) # 1.39

        self.log_odds_max = 5.0
        self.log_odds_min = -5.0

        # Particle filter
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.particles = [Particle(0.0, 0.0, 0.0, 1.0/self.num_particles, (self.map_height_cells, self.map_width_cells)) for _ in range(self.num_particles)]
        self.last_odom = None

        # ROS2 publishers/subscribers
        map_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', map_qos_profile)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_subscriber = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').get_parameter_value().string_value,
            self.odom_callback,
            10)
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').get_parameter_value().string_value,
            self.scan_callback,
            rclpy.qos.qos_profile_sensor_data)

        self.get_logger().info("Python SLAM node with particle filter initialized.")
        self.map_publish_timer = self.create_timer(1.0, self.publish_map)

    def odom_callback(self, msg: Odometry):
        # Store odometry for motion update
        self.last_odom = msg

    def scan_callback(self, msg: LaserScan): #ver parte chequear
        if self.last_odom is None:
            return

        # 1. Motion update (sample motion model)
        odom = self.last_odom
        # TODO: Retrieve odom_pose from odom message - remember that orientation is a quaternion
        odom_pose = odom.pose.pose
        odom_x = odom_pose.position.x
        odom_y = odom_pose.position.y
        odom_quat = odom_pose.orientation
        odom_theta = tf_transformations.euler_from_quaternion(
            [odom_quat.x, odom_quat.y, odom_quat.z, odom_quat.w])[2]  # Extract yaw angle
        self.get_logger().debug(f"Odometry received: x={odom_x}, y={odom_y}, theta={odom_theta}")   



        # TODO: Model the particles around the current pose
        for p in self.particles:
            # Add noise to simulate motion uncertainty
            noise_x = np.random.normal(0.0, 0.1)  # Standard deviation of 0.1 meters
            noise_y = np.random.normal(0.0, 0.1)  # Standard deviation of 0.1 meters
            noise_theta = np.random.normal(0.0, 0.1)  # Standard deviation of 0.1 radians
            p.x += odom_x + noise_x
            p.y += odom_y + noise_y
            p.theta = self.angle_diff(p.theta + odom_theta + noise_theta, 2 * np.pi)  # Normalize angle
        

        # TODO: 2. Measurement update (weight particles)
        weights = []
        for p in self.particles:
            weight = self.compute_weight(p, msg) # Compute weights for each particle
            # Save, append
            weights.append(weight)
        weights = np.array(weights)
        weights_sum = np.sum(weights)
        weights /= weights_sum

        self.get_logger().debug(f"Particle weights: {weights}")
        

        # Normalize weights

        for i, p in enumerate(self.particles):
            p.weight = weights[i] # Resave weights

        # 3. Resample
        self.particles = self.resample_particles(self.particles)

        # TODO: 4. Use weighted mean of all particles for mapping and pose (update current_map_pose and current_odom_pose, for each particle)
        # Calculate weighted mean pose from all particles
        total_weight = sum(p.weight for p in self.particles)

        # Weighted mean for x and y coordinates
        self.current_map_x = sum(p.x * p.weight for p in self.particles) / total_weight
        self.current_map_y = sum(p.y * p.weight for p in self.particles) / total_weight
        
        # For angles, we need to handle the circular nature (avoid averaging 359° and 1°)
        # Convert to unit vectors, average, then convert back
        cos_sum = sum(np.cos(p.theta) * p.weight for p in self.particles)
        sin_sum = sum(np.sin(p.theta) * p.weight for p in self.particles)
        self.current_map_theta = np.arctan2(sin_sum, cos_sum)

        self.get_logger().debug(f"Current map pose: x={self.current_map_x}, y={self.current_map_y}, theta={self.current_map_theta}")
        #CHEQUEAR QUE ONDA

        # 5. Mapping (update map with best particle's pose)
        for p in self.particles:
            self.update_map(p, msg)

        # 6. Broadcast map->odom transform
        self.broadcast_map_to_odom()

    def compute_weight(self, particle, scan_msg): #ver mensaje bauti
        # Simple likelihood: count how many endpoints match occupied cells
        score = 0.0
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        for i, range_dist in enumerate(scan_msg.ranges):
            if range_dist < scan_msg.range_min or range_dist > scan_msg.range_max or math.isnan(range_dist):
                continue
                
            # TODO: Compute the map coordinates of the endpoint: transform the scan into the map frame
            # Calculate the angle of this laser beam
            beam_angle = scan_msg.angle_min + i * scan_msg.angle_increment
            # Transform to global coordinates
            global_angle = robot_theta + beam_angle
            
            # Calculate endpoint in world coordinates
            endpoint_x = robot_x + range_dist * math.cos(global_angle)
            endpoint_y = robot_y + range_dist * math.sin(global_angle)
            
            # Convert to map cell coordinates
            map_x = int((endpoint_x - self.map_origin_x) / self.resolution)
            map_y = int((endpoint_y - self.map_origin_y) / self.resolution)


            
            # TODO: Use particle.log_odds_map for scoring
            # Check if the endpoint is within map bounds
            if 0 <= map_x < self.map_width_cells and 0 <= map_y < self.map_height_cells:
                # Get the log-odds value at this location
                log_odds_value = particle.log_odds_map[map_y, map_x]
                
                # Convert log-odds to probability: P = exp(log_odds) / (1 + exp(log_odds))
                # For numerical stability, use different formulas based on log_odds sign
                if log_odds_value > 0:
                    prob_occupied = log_odds_value / (1.0 + log_odds_value)
                else:
                    exp_log_odds = math.exp(log_odds_value)
                    prob_occupied = exp_log_odds / (1.0 + exp_log_odds)
                
                # Add to score - higher probability of occupation = higher score
                score += prob_occupied
            else:
                # Penalize points outside the map
                score -= 0.1


        return score + 1e-6

    def resample_particles(self, particles):
        # TODO: Resample particles
        # Extract weights from particles
        weights = np.array([p.weight for p in particles])
        
        # Sum the weights 
        sum_weights = np.sum(weights)    
        
        # Normalize the weights
        p = []
        for i in range(len(weights)):
            p.append(weights[i]/sum_weights)
        
        # Calculate the CDF of the weights
        pdf = []        
        for k in range(len(p)):
            if k == 0:
                pdf.append(p[k])
            else:
                pdf.append(pdf[k-1] + p[k])
            
        # Calculate the step for random sampling, depends on number of particles
        step = 1.0/len(p)       
        
        # Sample a value in between [0,step) uniformly
        seed = np.random.uniform(0, step)
        
        # Resample the particles based on the seed, step and calculated CDF
        new_particles = []
        i = 0
        for h in range(len(p)):
            u = seed + h * step
            while u > pdf[i]:
                i += 1
            # Create a copy of the selected particle
            selected_particle = particles[i]
            new_particle = Particle(selected_particle.x, selected_particle.y, selected_particle.theta, 
                                   1.0/len(particles), selected_particle.log_odds_map.shape)
            new_particle.log_odds_map = selected_particle.log_odds_map.copy()
            new_particles.append(new_particle)
        
        return new_particles

    def update_map(self, particle, scan_msg):
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        for i, range_dist in enumerate(scan_msg.ranges):
            is_hit = range_dist < scan_msg.range_max
            current_range = min(range_dist, scan_msg.range_max)
            if math.isnan(current_range) or current_range < scan_msg.range_min:
                continue
            # TODO: Update map: transform the scan into the map frame
            # Calculate the angle of this laser beam
            beam_angle = scan_msg.angle_min + i * scan_msg.angle_increment
            # Transform to global coordinates
            global_angle = robot_theta + beam_angle
            
            # Calculate endpoint in world coordinates
            endpoint_x = robot_x + current_range * math.cos(global_angle)
            endpoint_y = robot_y + current_range * math.sin(global_angle)
            
            # Convert robot position to map coordinates
            robot_map_x = int((robot_x - self.map_origin_x) / self.resolution)
            robot_map_y = int((robot_y - self.map_origin_y) / self.resolution)
            
            # Convert endpoint to map coordinates
            endpoint_map_x = int((endpoint_x - self.map_origin_x) / self.resolution)
            endpoint_map_y = int((endpoint_y - self.map_origin_y) / self.resolution)

            # TODO: Use self.bresenham_line for free cells
            # Mark all cells along the ray as free (from robot to just before endpoint)
            self.bresenham_line(particle, robot_map_x, robot_map_y, endpoint_map_x, endpoint_map_y)

            # TODO: Update particle.log_odds_map accordingly
            # Mark the endpoint as occupied if we hit something
            if is_hit and 0 <= endpoint_map_x < self.map_width_cells and 0 <= endpoint_map_y < self.map_height_cells:
                particle.log_odds_map[endpoint_map_y, endpoint_map_x] += self.log_odds_occupied
                particle.log_odds_map[endpoint_map_y, endpoint_map_x] = np.clip(
                    particle.log_odds_map[endpoint_map_y, endpoint_map_x], 
                    self.log_odds_min, 
                    self.log_odds_max
                )

            

    def bresenham_line(self, particle, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        path_len = 0
        max_path_len = dx + dy
        while not (x0 == x1 and y0 == y1) and path_len < max_path_len:
            if 0 <= x0 < self.map_width_cells and 0 <= y0 < self.map_height_cells:
                particle.log_odds_map[y0, x0] += self.log_odds_free
                particle.log_odds_map[y0, x0] = np.clip(particle.log_odds_map[y0, x0], self.log_odds_min, self.log_odds_max)
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            path_len += 1

    def publish_map(self):
        # TODO: Fill in map_msg fields and publish one map
        map_msg = OccupancyGrid()
        
        # Header information
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
        
        # Map metadata
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.map_width_cells
        map_msg.info.height = self.map_height_cells
        map_msg.info.origin.position.x = self.map_origin_x
        map_msg.info.origin.position.y = self.map_origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0  # No rotation
        
        # Create combined map from all particles (use best particle or average)
        # For simplicity, use the particle with highest weight
        if self.particles:
            best_particle = max(self.particles, key=lambda p: p.weight)
            
            # Convert log-odds to occupancy grid values [0-100, -1 for unknown]
            map_data = []
            for y in range(self.map_height_cells):
                for x in range(self.map_width_cells):
                    log_odds = best_particle.log_odds_map[y, x]
                    
                    # Convert log-odds to probability
                    if abs(log_odds) < 0.1:  # Unknown threshold
                        occupancy = -1  # Unknown
                    else:
                        # Convert to probability: P = exp(log_odds) / (1 + exp(log_odds))
                        if log_odds > 0:
                            prob = log_odds / (1.0 + log_odds)
                        else:
                            exp_log_odds = math.exp(log_odds)
                            prob = exp_log_odds / (1.0 + exp_log_odds)
                        
                        # Scale to 0-100 range
                        occupancy = int(prob * 100)
                        occupancy = max(0, min(100, occupancy))
                    
                    map_data.append(occupancy)
            
            map_msg.data = map_data
        
        self.map_publisher.publish(map_msg)
        self.get_logger().debug("Map published.")

    def broadcast_map_to_odom(self):
        # TODO: Broadcast map->odom transform
        t = TransformStamped()
        
        # Header
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
        t.child_frame_id = self.get_parameter('odom_frame').get_parameter_value().string_value
        
        # Check if we have current pose estimate
        if hasattr(self, 'current_map_x') and hasattr(self, 'current_map_y') and hasattr(self, 'current_map_theta'):
            # Get current odometry pose
            if self.last_odom is not None:
                odom_pose = self.last_odom.pose.pose
                odom_x = odom_pose.position.x
                odom_y = odom_pose.position.y
                odom_quat = odom_pose.orientation
                odom_theta = tf_transformations.euler_from_quaternion(
                    [odom_quat.x, odom_quat.y, odom_quat.z, odom_quat.w])[2]
                
                # Calculate map->odom transform
                # map_pose = map_to_odom * odom_pose
                # Therefore: map_to_odom = map_pose * odom_pose^(-1)
                
                # Calculate the transform from map to odom frame
                delta_x = self.current_map_x - odom_x
                delta_y = self.current_map_y - odom_y
                delta_theta = self.angle_diff(self.current_map_theta, odom_theta)
                
                # Set translation
                t.transform.translation.x = delta_x
                t.transform.translation.y = delta_y
                t.transform.translation.z = 0.0
                
                # Convert angle to quaternion
                quaternion = tf_transformations.quaternion_from_euler(0, 0, delta_theta)
                t.transform.rotation.x = quaternion[0]
                t.transform.rotation.y = quaternion[1]
                t.transform.rotation.z = quaternion[2]
                t.transform.rotation.w = quaternion[3]
            else:
                # No odometry available, set identity transform
                t.transform.translation.x = 0.0
                t.transform.translation.y = 0.0
                t.transform.translation.z = 0.0
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = 1.0
        else:
            # No pose estimate available, set identity transform
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(t)

    @staticmethod
    def angle_diff(a, b):
        d = a - b
        while d > np.pi:
            d -= 2 * np.pi
        while d < -np.pi:
            d += 2 * np.pi
        return d

def main(args=None):
    rclpy.init(args=args)
    node = PythonSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()