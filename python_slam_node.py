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

        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_width_meters', 5.0)
        self.declare_parameter('map_height_meters', 5.0)
        self.declare_parameter('num_particles', 10)

        self.resolution = self.get_parameter('map_resolution').get_parameter_value().double_value
        self.map_width_m = self.get_parameter('map_width_meters').get_parameter_value().double_value
        self.map_height_m = self.get_parameter('map_height_meters').get_parameter_value().double_value
        self.map_width_cells = int(self.map_width_m / self.resolution)
        self.map_height_cells = int(self.map_height_m / self.resolution)
        self.map_origin_x = -self.map_width_m / 2.0
        self.map_origin_y = -5.0

        # Adjusted log-odds values for better persistence
        self.log_odds_free = -0.4      # Less aggressive free space updates
        self.log_odds_occupied = 2.0   # Strong evidence for obstacles
        
        self.log_odds_max = 5.0# Higher max for better persistence
        self.log_odds_min = -5.0
        
        self.occupied_threshold = 2.0# Lower threshold for occupied
        self.free_threshold = -1.0   

        # Particle filter
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.particles = [Particle(0.0, 0.0, 0.0, 1.0/self.num_particles, 
                                 (self.map_height_cells, self.map_width_cells)) 
                         for _ in range(self.num_particles)]
        self.last_odom = None
        self.prev_odom = None
        
        # Initialize pose estimates
        self.current_map_x = 0.0
        self.current_map_y = 0.0  
        self.current_map_theta = 0.0

        #Initialize angular velocity and rotation parameters
        self.last_angular_vel = 0.0
        self.is_rotating = False
        self.rotation_threshold = 0.1
        self.rotation_noise_scale = 2.0
        self.rotation_update_scale = 0.5

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

        self.get_logger().info("=== Python SLAM Node Initialized ===")
        self.get_logger().info(f"Map parameters: {self.map_width_cells}x{self.map_height_cells} cells")
        self.get_logger().info(f"Number of particles: {self.num_particles}")
        self.map_publish_timer = self.create_timer(1.0, self.publish_map)
        
        self.first_scan_processed = False

    def odom_callback(self, msg: Odometry):
        self.prev_odom = self.last_odom
        self.last_odom = msg

        #Track angular velocity
        self.last_angular_vel = msg.twist.twist.angular.z
        self.is_rotating = abs(self.last_angular_vel) > self.rotation_threshold

    def scan_callback(self, msg: LaserScan):
        if self.last_odom is None:
            self.get_logger().warn("No odometry data available yet")
            return
        
            # Log rotation status
        if self.is_rotating:
            self.get_logger().debug(f"Robot rotating at {self.last_angular_vel:.2f} rad/s")

        # 1. Motion update
        odom = self.last_odom
        odom_pose = odom.pose.pose
        odom_x = odom_pose.position.x
        odom_y = odom_pose.position.y
        odom_quat = odom_pose.orientation
        odom_theta = tf_transformations.euler_from_quaternion(
            [odom_quat.x, odom_quat.y, odom_quat.z, odom_quat.w])[2]

        if self.prev_odom is not None:
            # Calculate motion since last update
            prev_pose = self.prev_odom.pose.pose
            prev_x = prev_pose.position.x
            prev_y = prev_pose.position.y
            prev_quat = prev_pose.orientation
            prev_theta = tf_transformations.euler_from_quaternion(
                [prev_quat.x, prev_quat.y, prev_quat.z, prev_quat.w])[2]
            
            # Motion deltas
            delta_x = odom_x - prev_x
            delta_y = odom_y - prev_y
            delta_theta = self.angle_diff(odom_theta, prev_theta)
            
            # Update particles with motion
            for p in self.particles:
                noise_x = np.random.normal(0.0, 0.02)
                noise_y = np.random.normal(0.0, 0.02)  
                noise_theta = np.random.normal(0.0, 0.02)  
                
                p.x += delta_x + noise_x
                p.y += delta_y + noise_y
                p.theta = self.angle_diff(p.theta + delta_theta + noise_theta, 0.0)
        else:
            # First scan - initialize particles
            for p in self.particles:
                noise_x = np.random.normal(0.0, 0.1)
                noise_y = np.random.normal(0.0, 0.1)  
                noise_theta = np.random.normal(0.0, 0.1)  
                p.x = odom_x + noise_x
                p.y = odom_y + noise_y
                p.theta = odom_theta + noise_theta

        # 2. Update maps for all particles
        for p in self.particles:
            self.update_map(p, msg)

        # 3. Compute weights
        weights = []
        for i, p in enumerate(self.particles):
            weight = self.compute_weight(p, msg)
            weights.append(weight)
        
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        else:
            weights.fill(1.0 / len(weights))

        for i, p in enumerate(self.particles):
            p.weight = weights[i]

        # 4. Resample
        self.particles = self.resample_particles(self.particles)

        # 5. Calculate weighted mean pose
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            self.current_map_x = sum(p.x * p.weight for p in self.particles) / total_weight
            self.current_map_y = sum(p.y * p.weight for p in self.particles) / total_weight
            
            cos_sum = sum(np.cos(p.theta) * p.weight for p in self.particles) / total_weight
            sin_sum = sum(np.sin(p.theta) * p.weight for p in self.particles) / total_weight
            self.current_map_theta = np.arctan2(sin_sum, cos_sum)

        # 6. Broadcast transform
        self.broadcast_map_to_odom()

    def compute_weight(self, particle, scan_msg):
        score = 0.0
        valid_measurements = 0
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        
        for i, range_dist in enumerate(scan_msg.ranges):
            if range_dist < scan_msg.range_min or range_dist > scan_msg.range_max or math.isnan(range_dist):
                continue
                
            beam_angle = scan_msg.angle_min + i * scan_msg.angle_increment
            global_angle = robot_theta + beam_angle
            
            endpoint_x = robot_x + range_dist * math.cos(global_angle)
            endpoint_y = robot_y + range_dist * math.sin(global_angle)
            
            map_x = int((endpoint_x - self.map_origin_x) / self.resolution)
            map_y = int((endpoint_y - self.map_origin_y) / self.resolution)

            if 0 <= map_x < self.map_width_cells and 0 <= map_y < self.map_height_cells:
                valid_measurements += 1
                log_odds_value = particle.log_odds_map[map_y, map_x]
                
                if abs(log_odds_value) < 0.01:
                    prob_occupied = 0.5
                else:
                    prob_occupied = 1.0 / (1.0 + math.exp(-log_odds_value))
                
                score += prob_occupied

        if valid_measurements > 0:
            score /= valid_measurements
        
        return max(score + 1e-6, 1e-6)

    def resample_particles(self, particles):
        """Fixed resampling that properly copies particle maps"""
        weights = np.array([p.weight for p in particles])
        
        if np.sum(weights) == 0:
            return particles
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Cumulative distribution
        cumulative_sum = np.cumsum(weights)
        
        # Low variance resampling
        new_particles = []
        r = np.random.uniform(0, 1.0/len(particles))
        
        for m in range(len(particles)):
            u = r + m * (1.0/len(particles))
            i = np.searchsorted(cumulative_sum, u)
            i = min(i, len(particles) - 1)  # Ensure valid index
            
            # Create a deep copy of the selected particle INCLUDING THE MAP
            selected_particle = particles[i]
            new_particle = Particle(
                selected_particle.x, 
                selected_particle.y, 
                selected_particle.theta,
                1.0/len(particles),
                selected_particle.log_odds_map.shape
            )
            # CRITICAL: Copy the map data
            new_particle.log_odds_map = selected_particle.log_odds_map.copy()
            new_particles.append(new_particle)
        
        return new_particles

    def update_map(self, particle, scan_msg):
        robot_x, robot_y, robot_theta = particle.x, particle.y, particle.theta
        updates_made = 0
        obstacles_detected = 0
        
        # Store original values
        original_log_odds_free = self.log_odds_free
        original_log_odds_occupied = self.log_odds_occupied
        
        # Temporarily modify the instance variables if rotating
        if hasattr(self, 'is_rotating') and self.is_rotating:
            self.log_odds_free = self.log_odds_free * 0.5
            self.log_odds_occupied = self.log_odds_occupied * 0.5
            self.get_logger().debug("Using reduced update rates during rotation")

        # Convert robot position to map coordinates once
        robot_map_x = int((robot_x - self.map_origin_x) / self.resolution)
        robot_map_y = int((robot_y - self.map_origin_y) / self.resolution)
        
        self.get_logger().debug(f"Updating map for particle at ({robot_x:.2f}, {robot_y:.2f}, {robot_theta:.2f})")
        self.get_logger().debug(f"Robot map coordinates: ({robot_map_x}, {robot_map_y})")
        
        for i, range_dist in enumerate(scan_msg.ranges):
            # Skip invalid measurements
            if math.isnan(range_dist) or range_dist < scan_msg.range_min:
                continue
                
            # Determine if this is a hit (obstacle detected) or max range
            is_hit = range_dist < scan_msg.range_max
            current_range = min(range_dist, scan_msg.range_max)
            
            # TODO: Update map: transform the scan into the map frame
            # Calculate the angle of this laser beam
            beam_angle = scan_msg.angle_min + i * scan_msg.angle_increment
            # Transform to global coordinates
            global_angle = robot_theta + beam_angle
            
            # Calculate endpoint in world coordinates
            endpoint_x = robot_x + current_range * math.cos(global_angle)
            endpoint_y = robot_y + current_range * math.sin(global_angle)
            
            # Convert endpoint to map coordinates
            endpoint_map_x = int((endpoint_x - self.map_origin_x) / self.resolution)
            endpoint_map_y = int((endpoint_y - self.map_origin_y) / self.resolution)

            # TODO: Use self.bresenham_line for free cells
            # Mark all cells along the ray as free (from robot to just before endpoint)
            self.bresenham_line(particle, robot_map_x, robot_map_y, endpoint_map_x, endpoint_map_y)

            # TODO: Update particle.log_odds_map accordingly
            # Mark the endpoint as occupied if we hit something within bounds
            if is_hit and 0 <= endpoint_map_x < self.map_width_cells and 0 <= endpoint_map_y < self.map_height_cells:
                old_value = particle.log_odds_map[endpoint_map_y, endpoint_map_x]
                particle.log_odds_map[endpoint_map_y, endpoint_map_x] += self.log_odds_occupied
                particle.log_odds_map[endpoint_map_y, endpoint_map_x] = np.clip(
                    particle.log_odds_map[endpoint_map_y, endpoint_map_x], 
                    self.log_odds_min, 
                    self.log_odds_max
                )
                new_value = particle.log_odds_map[endpoint_map_y, endpoint_map_x]
                updates_made += 1
                obstacles_detected += 1
                
                # Log significant obstacle detections
                if abs(old_value - new_value) > 0.1:
                    self.get_logger().debug(f"Obstacle detected at ({endpoint_map_x},{endpoint_map_y}): {old_value:.2f} -> {new_value:.2f}")

        if updates_made > 0:
            self.get_logger().info(f"Map updated: {obstacles_detected} obstacles detected, {updates_made} total updates")
        else:
            self.get_logger().debug("No map updates made this scan")

        # Restore original values
        self.log_odds_free = original_log_odds_free
        self.log_odds_occupied = original_log_odds_occupied

    def bresenham_line(self, particle, x0, y0, x1, y1):
        """Draw a line using Bresenham's algorithm and mark cells as free."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        path_len = 0
        max_path_len = dx + dy + 10  # Allow a bit more leeway
        cells_updated = 0
        
        # Stop one cell before the endpoint to avoid marking the obstacle as free
        while path_len < max_path_len:
            # Check if we're at the endpoint (stop before marking it)
            if x0 == x1 and y0 == y1:
                break
                
            # Check bounds before updating
            if 0 <= x0 < self.map_width_cells and 0 <= y0 < self.map_height_cells:
                # Only update if we're not too close to the endpoint
                distance_to_end = abs(x1 - x0) + abs(y1 - y0)
                if distance_to_end > 1:  # Don't mark the cell right next to obstacle as free
                    old_value = particle.log_odds_map[y0, x0]
                    particle.log_odds_map[y0, x0] += self.log_odds_free
                    particle.log_odds_map[y0, x0] = np.clip(
                        particle.log_odds_map[y0, x0], 
                        self.log_odds_min, 
                        self.log_odds_max
                    )
                    cells_updated += 1
                    
                    # Log significant changes
                    if abs(old_value - particle.log_odds_map[y0, x0]) > 0.1:
                        self.get_logger().debug(f"Free cell update at ({x0},{y0}): {old_value:.2f} -> {particle.log_odds_map[y0, x0]:.2f}")
            
            # Move to next cell
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
            path_len += 1
            
        if cells_updated > 0:
            self.get_logger().debug(f"Bresenham line updated {cells_updated} free cells")

    def log_odds_to_occupancy(self, log_odds):
        """
        Convert log-odds to occupancy value for ROS OccupancyGrid.
        
        Args:
            log_odds: Log-odds value from the map
            
        Returns:
            int: Occupancy value (0-100) or -1 for unknown
        """
        # Check if cell has never been observed
        if abs(log_odds) < 0.01:
            return -1  # Unknown
        
        # Convert log-odds to probability
        # Using the logistic function: p = 1 / (1 + exp(-log_odds))
        probability = 1.0 / (1.0 + math.exp(-log_odds))
        
        # Convert probability [0,1] to occupancy value [0,100]
        occupancy = int(probability * 100)
        
        # Clamp to valid range
        return max(0, min(100, occupancy))


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
                    
                    occupancy = self.log_odds_to_occupancy(log_odds)
                    map_data.append(occupancy)
            
            map_msg.data = map_data
            
            self.get_logger().info(f"Map published")
            self.get_logger().debug(f"Best particle weight: {best_particle.weight:.6f}")
            
        else:
            # No particles, publish empty map
            map_data = [-1] * (self.map_width_cells * self.map_height_cells)
            map_msg.data = map_data
            self.get_logger().warn("No particles available, publishing empty map")
        
        self.map_publisher.publish(map_msg)

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
        """Calculate the difference between two angles, handling wrap-around"""
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

