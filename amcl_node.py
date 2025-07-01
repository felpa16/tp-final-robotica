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
        self.declare_parameter('laser_max_range', 2.5)
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('obstacle_detection_distance', 0.1)  # Equilibrio entre seguridad y movilidad
        self.declare_parameter('obstacle_avoidance_turn_speed', 0.25)  # Giro un poco más rápido

        # --- Parameters to set ---
        # Parámetros optimizados para navegación lenta y precisa
        self.declare_parameter('num_particles', 600)  # Reducido para mejor rendimiento
        self.declare_parameter('alpha1', 0.1)  # Ruido rotacional inicial reducido
        self.declare_parameter('alpha2', 0.1)  # Ruido rotacional vs traslación
        self.declare_parameter('alpha3', 0.1)  # Ruido traslacional reducido
        self.declare_parameter('alpha4', 0.1)  # Ruido traslacional vs rotación
        self.declare_parameter('z_hit', 0.8)  # Mayor peso al modelo de medición directa
        self.declare_parameter('z_rand', 0.2)  # Menor peso al ruido
        self.declare_parameter('lookahead_distance', 0.5)  # Distancia más corta para mayor precisión
        self.declare_parameter('linear_velocity', 0.1)  # Velocidad mucho más lenta
        self.declare_parameter('goal_tolerance', 0.12)  # Tolerancia más precisa
        self.declare_parameter('path_pruning_distance', 0.15)  # Suavizado más conservador  
        self.declare_parameter('safety_margin_cells', 3)  # Mayor margen de seguridad - Este valor se puede ajustar según el tamaño del robot - Mientras mas 

        
        
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
        self.particles_initialized = False
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
        
        # Log frame names for debugging
        self.get_logger().info(f'Frame configuration: map={self.map_frame_id}, odom={self.odom_frame_id}, base={self.base_frame_id}')
        self.get_logger().info(f'Topics: scan={self.get_parameter("scan_topic").value}, goal={self.get_parameter("goal_topic").value}, cmd_vel={self.get_parameter("cmd_vel_topic").value}')
        self.get_logger().info(f'Navigation params: linear_vel={self.linear_velocity:.2f}, obstacle_dist={self.obstacle_detection_distance:.2f}, safety_margin={self.safety_margin_cells}')
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
        self.particles_initialized = True
        self.last_odom_pose = None # Reset odom tracking

    def initialize_particles(self, initial_pose):
        # TODO: Inicializar particulas en base a la pose inicial con variaciones aleatorias
        # Deben ser la misma cantidad de particulas que self.num_particles
        # Deben tener un peso
        
        # Extraer pose inicial
        x = initial_pose.position.x
        y = initial_pose.position.y
        q = initial_pose.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        
        # Generar partículas con ruido gaussiano alrededor de la pose inicial
        noise_x = np.random.normal(0, 0.5, self.num_particles)
        noise_y = np.random.normal(0, 0.5, self.num_particles)
        noise_yaw = np.random.normal(0, np.pi/4, self.num_particles)
        
        self.particles[:, 0] = x + noise_x
        self.particles[:, 1] = y + noise_y
        self.particles[:, 2] = yaw + noise_yaw
        
        # Normalizar ángulos
        self.particles[:, 2] = np.arctan2(np.sin(self.particles[:, 2]), np.cos(self.particles[:, 2]))
        
        # Peso uniforme para todas las partículas
        self.weights = np.ones(self.num_particles) / self.num_particles

        self.publish_particles()

    def initialize_particles_randomly(self):
        # TODO: Inizializar particulas aleatoriamente en todo el mapa
        
        # Obtener celdas libres del mapa
        free_cells = np.where(self.grid == 0)  # 0 = espacio libre
        if len(free_cells[0]) == 0:
            self.get_logger().warn("No free cells found in map for random initialization")
            return
            
        # Seleccionar posiciones aleatorias de las celdas libres
        num_free = len(free_cells[0])
        selected_indices = np.random.choice(num_free, self.num_particles, replace=True)
        
        # Convertir coordenadas de grid a world
        for i in range(self.num_particles):
            idx = selected_indices[i]
            # CORRECCIÓN: Invertir el orden de los índices
            gy = free_cells[0][idx]  # fila (y)
            gx = free_cells[1][idx]  # columna (x)
            wx, wy = self.grid_to_world(gx, gy)
            
            self.particles[i, 0] = wx
            self.particles[i, 1] = wy
            self.particles[i, 2] = np.random.uniform(-np.pi, np.pi)  # Usar rango completo
            
        # Peso uniforme para todas las partículas
        self.weights = np.ones(self.num_particles) / self.num_particles

        self.publish_particles()

    def timer_callback(self):
        # TODO: Implementar maquina de estados para cada caso.
        # Debe haber estado para PLANNING, NAVIGATING y AVOIDING_OBSTACLE, pero pueden haber más estados si se desea.
        if not self.map_received:
            return

        # --- Bloque 1: Localización (siempre ejecutar) ---
        if self.latest_scan is None:
            return
            
        if not self.particles_initialized:
            self.initialize_particles_randomly()
            self.particles_initialized = True
            return

        # Obtener transformación de odometría (puede fallar)
        current_odom_tf = self.get_odom_transform()
        tf_available = current_odom_tf is not None

        # Aplicar modelo de movimiento solo si hay odometría válida
        if tf_available:
            self.motion_model(current_odom_tf)
        
        # Aplicar modelo de medición (siempre)
        self.measurement_model()
        
        # Resamplear partículas (siempre)
        self.resample()
        
        # Estimar pose actual (siempre)
        estimated_pose = self.estimate_pose()

        # Publicar pose estimada y partículas (siempre)
        self.publish_pose(estimated_pose)
        self.publish_particles()
        
        # Publicar transformación solo si hay odometría válida
        if tf_available:
            self.publish_transform(estimated_pose, current_odom_tf)
        
        # --- Bloque 2: Máquina de Estados ---
        current_pose_tuple = (estimated_pose.position.x, estimated_pose.position.y)

        if self.state == State.IDLE:
            # Esperar goal
            self.stop_robot()
            
        elif self.state == State.PLANNING:
            # Planificar ruta
            if self.goal_pose is not None:
                self.get_logger().info("Planning path to goal...")
                start_pos = current_pose_tuple
                goal_pos = (self.goal_pose.position.x, self.goal_pose.position.y)
                # Planificar, densificar SIEMPRE con muchos puntos, suavizar y publicar el camino
                path = self.a_star_planning(start_pos, goal_pos)
                if path:
                    # Densificar antes y después de suavizar para asegurar muchos puntos
                    densified_path = self.densify_path(path, spacing=0.01, max_points=2000)
                    smoothed_path = self.smooth_path(densified_path)
                    final_path = self.densify_path(smoothed_path, spacing=0.01, max_points=2000)
                    self.current_path = final_path
                    # Convertir path a mensaje Path y publicar
                    path_msg = Path()
                    path_msg.header.stamp = self.get_clock().now().to_msg()
                    path_msg.header.frame_id = self.map_frame_id
                    for p in self.current_path:
                        pose_stamped = PoseStamped()
                        pose_stamped.header.stamp = path_msg.header.stamp
                        pose_stamped.header.frame_id = self.map_frame_id
                        pose_stamped.pose.position.x = p[0]
                        pose_stamped.pose.position.y = p[1]
                        path_msg.poses.append(pose_stamped)
                    self.path_pub.publish(path_msg)
                    self.get_logger().info(f"Path found with {len(self.current_path)} points. State -> NAVIGATING")
                    self.state = State.NAVIGATING
                else:
                    self.get_logger().error("Failed to find a path to the goal.")
                    self.state = State.IDLE
                    self.goal_pose = None # Reset goal
                
        elif self.state == State.NAVIGATING:
            # Seguir trayectoria
            if self.check_for_obstacles():
                self.get_logger().warn("Obstacle detected! State -> AVOIDING_OBSTACLE")
                self.stop_robot()
                self.state = State.AVOIDING_OBSTACLE
            elif self.current_path:
                # Verificar si llegamos al goal
                goal_pos = np.array(self.current_path[-1])
                dist_to_goal = np.linalg.norm(np.array(current_pose_tuple) - goal_pos)
                
                if dist_to_goal < self.goal_tolerance:
                    self.get_logger().info("Goal reached! Stopping robot.")
                    # Enviar comando de parada explícito
                    cmd_stop = Twist()
                    cmd_stop.linear.x = 0.0
                    cmd_stop.angular.z = 0.0
                    self.cmd_vel_pub.publish(cmd_stop)
                    self.get_logger().info(f"[STOP] cmd_vel → lin={cmd_stop.linear.x:.2f}, ang={cmd_stop.angular.z:.2f}")
                    
                    # Cambiar estado
                    self.state = State.IDLE
                    self.current_path = None
                    self.goal_pose = None
                else:
                    # Seguir la ruta con Pure Pursuit
                    cmd_vel = self.pure_pursuit_control(estimated_pose, self.current_path)
                    self.get_logger().info(f"[NAV] cmd_vel → lin={cmd_vel.linear.x:.2f}, ang={cmd_vel.angular.z:.2f}")
                    self.cmd_vel_pub.publish(cmd_vel)
            else:
                # No hay path, volver a IDLE
                self.state = State.IDLE
            
        elif self.state == State.AVOIDING_OBSTACLE:
            # Girar 15° hacia el lado con más espacio libre, sin replanning
            if not self.check_for_obstacles():
                self.get_logger().info("Obstacle cleared, resuming navigation...")
                self.stop_robot()  # Detener el robot antes de navegar
                self.state = State.NAVIGATING
            else:
                direction = self.get_obstacle_avoidance_direction()
                cmd = Twist()
                cmd.linear.x = 0.0
                # Girar 15° (0.2618 rad) en el sentido elegido, velocidad angular fija
                cmd.angular.z = direction * 0.2618  # 15 grados/seg
                sentido = "izquierda" if direction == 1 else "derecha"
                self.get_logger().info(f"[AVOID] cmd_vel → lin={{cmd.linear.x:.2f}}, ang={{cmd.angular.z:.2f}} (giro a {{sentido}} 15°)")
                self.cmd_vel_pub.publish(cmd)


    def get_odom_transform(self):
        try:
            # Usar tiempo más reciente disponible para evitar warnings de TF_OLD_DATA
            return self.tf_buffer.lookup_transform(
                self.odom_frame_id, 
                self.base_frame_id, 
                rclpy.time.Time(),  # Usar el tiempo más reciente
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f'Could not get transform from {self.odom_frame_id} to {self.base_frame_id}. Error: {e}', throttle_duration_sec=2.0)
            return None

    def motion_model(self, current_odom_tf):
        current_odom_pose = current_odom_tf.transform
        
        # TODO: Implementar el modelo de movimiento para actualizar las particulas.
        
        if self.last_odom_pose is None:
            self.last_odom_pose = current_odom_pose
            return
            
        # Calcular cambio en odometría
        dx = current_odom_pose.translation.x - self.last_odom_pose.translation.x
        dy = current_odom_pose.translation.y - self.last_odom_pose.translation.y
        
        # Convertir cuaterniones a yaw
        def quat_to_yaw(q):
            return np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        
        current_yaw = quat_to_yaw(current_odom_pose.rotation)
        last_yaw = quat_to_yaw(self.last_odom_pose.rotation)
        dtheta = current_yaw - last_yaw
        
        # Normalizar ángulo
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        
        # Calcular distancia y rotaciones
        trans = np.sqrt(dx**2 + dy**2)
        rot1 = np.arctan2(dy, dx) - last_yaw if trans > 1e-6 else 0.0
        rot2 = dtheta - rot1
        
        # Normalizar ángulos
        rot1 = np.arctan2(np.sin(rot1), np.cos(rot1))
        rot2 = np.arctan2(np.sin(rot2), np.cos(rot2))
        
        # Aplicar modelo de movimiento con ruido a cada partícula
        for i in range(self.num_particles):
            # Ruido proporcional al movimiento
            rot1_noise = rot1 + np.random.normal(0, self.alphas[0] * abs(rot1) + self.alphas[1] * trans)
            trans_noise = trans + np.random.normal(0, self.alphas[2] * trans + self.alphas[3] * (abs(rot1) + abs(rot2)))
            rot2_noise = rot2 + np.random.normal(0, self.alphas[0] * abs(rot2) + self.alphas[1] * trans)
            
            # Actualizar partícula
            self.particles[i, 0] += trans_noise * np.cos(self.particles[i, 2] + rot1_noise)
            self.particles[i, 1] += trans_noise * np.sin(self.particles[i, 2] + rot1_noise)
            self.particles[i, 2] += rot1_noise + rot2_noise
            
            # Normalizar ángulo
            self.particles[i, 2] = np.arctan2(np.sin(self.particles[i, 2]), np.cos(self.particles[i, 2]))
        
        self.last_odom_pose = current_odom_pose

    def measurement_model(self):
        map_res = self.map_data.info.resolution
        map_origin = self.map_data.info.origin.position
        map_w = self.map_data.info.width
        map_h = self.map_data.info.height
        map_img = np.array(self.map_data.data).reshape((map_h, map_w))

        # TODO: Implementar el modelo de medición para actualizar los pesos de las particulas por particula
        
        # Obtener datos del scan
        ranges = np.array(self.latest_scan.ranges)
        angle_min = self.latest_scan.angle_min
        angle_increment = self.latest_scan.angle_increment
        range_max = min(self.latest_scan.range_max, self.laser_max_range)
        
        # Filtrar lecturas válidas
        valid_ranges = (ranges > self.latest_scan.range_min) & (ranges < range_max) & np.isfinite(ranges)
        
        # Submuestrear para eficiencia (cada 10 rayos)
        step = 10
        valid_indices = np.where(valid_ranges)[0][::step]
        
        if len(valid_indices) == 0:
            return
            
        # Para cada partícula, calcular likelihood
        for i in range(self.num_particles):
            px, py, ptheta = self.particles[i]
            log_likelihood = 0.0  # Usar log-likelihood para estabilidad numérica
            
            for idx in valid_indices:
                # Ángulo del rayo en frame del robot
                ray_angle = angle_min + idx * angle_increment
                
                # Ángulo absoluto en mapa
                abs_angle = ptheta + ray_angle
                abs_angle = np.arctan2(np.sin(abs_angle), np.cos(abs_angle)) # Normalizar
                
                # Ray casting para encontrar distancia esperada
                expected_range = self.ray_cast(px, py, abs_angle, range_max, map_img, map_res, map_origin)
                
                # Distancia observada
                observed_range = ranges[idx]
                
                # Modelo de sensor: combinación de hit + random
                # Probabilidad de un hit (gaussiana)
                sigma_hit = 0.2
                if expected_range > 0:  # Evitar división por cero
                    hit_prob = np.exp(-0.5 * ((observed_range - expected_range)**2) / (sigma_hit**2))
                    hit_prob /= (sigma_hit * np.sqrt(2 * np.pi))  # Normalización gaussiana
                else:
                    hit_prob = 1e-9
                
                rand_prob = 1.0 / range_max
                
                # Likelihood total para este rayo
                ray_likelihood = self.z_hit * hit_prob + self.z_rand * rand_prob
                
                # Asegurar que likelihood esté acotado
                ray_likelihood = np.clip(ray_likelihood, 1e-9, 1.0)
                log_likelihood += np.log(ray_likelihood)
            
            self.weights[i] = log_likelihood
        
        # Normalizar pesos (convirtiendo de log a lineal)
        if len(self.weights) == 0 or np.all(np.isinf(self.weights)):
            self.weights = np.ones(self.num_particles) / self.num_particles
            return
            
        max_log_weight = np.max(self.weights)
        self.weights = np.exp(self.weights - max_log_weight)
        
        weight_sum = np.sum(self.weights)
        if weight_sum > 1e-9:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def ray_cast(self, x, y, angle, max_range, map_img, resolution, origin):
        """Ray casting para encontrar la distancia al obstáculo más cercano"""
        step_size = resolution / 2.0  # Paso pequeño para precisión
        distance = 0.0
        
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        current_x = x
        current_y = y
        
        while distance < max_range:
            # Convertir a coordenadas de grid
            gx = int((current_x - origin.x) / resolution)
            gy = int((current_y - origin.y) / resolution)
            
            # Verificar límites del mapa
            if gx < 0 or gx >= map_img.shape[1] or gy < 0 or gy >= map_img.shape[0]:
                return max_range
                
            # Verificar si hay obstáculo (valor > 50 en occupancy grid)
            if map_img[gy, gx] > 50:
                return distance
                
            # Avanzar
            current_x += dx
            current_y += dy
            distance += step_size
            
        return max_range

    def resample(self):
        # TODO: Implementar el resampleo de las particulas basado en los pesos.
        
        # Low variance resampling
        new_particles = np.zeros_like(self.particles)
        
        # Calcular pesos acumulativos
        cumulative_weights = np.cumsum(self.weights)
        
        # Punto de inicio aleatorio
        r = np.random.uniform(0, 1.0 / self.num_particles)
        
        i = 0
        for m in range(self.num_particles):
            u = r + m / self.num_particles
            
            # Encontrar índice de partícula a copiar
            while u > cumulative_weights[i]:
                i += 1
                
            new_particles[m] = self.particles[i].copy()
        
        # Actualizar partículas y resetear pesos
        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_pose(self):
        # TODO: Implementar la estimación de pose a partir de las particulas y sus pesos.
        
        # Media ponderada de posición
        x = np.sum(self.particles[:, 0] * self.weights)
        y = np.sum(self.particles[:, 1] * self.weights)
        
        # Para el ángulo, usar la media circular
        sin_sum = np.sum(np.sin(self.particles[:, 2]) * self.weights)
        cos_sum = np.sum(np.cos(self.particles[:, 2]) * self.weights)
        theta = np.arctan2(sin_sum, cos_sum)
        
        # Crear mensaje Pose
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        
        # Convertir yaw a quaternion
        q = R.from_euler('z', theta).as_quat()
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
        try:
            map_to_base_mat = self.pose_to_matrix(estimated_pose)
            odom_to_base_mat = self.transform_to_matrix(odom_tf.transform)
            
            # Verificar que las matrices sean válidas
            if np.any(np.isnan(map_to_base_mat)) or np.any(np.isnan(odom_to_base_mat)):
                self.get_logger().warn("Invalid transformation matrix detected")
                return
                
            map_to_odom_mat = np.dot(map_to_base_mat, np.linalg.inv(odom_to_base_mat))
            
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.map_frame_id
            t.child_frame_id = self.odom_frame_id
            
            translation = map_to_odom_mat[:3, 3]
            rotation_matrix = map_to_odom_mat[:3, :3]
            
            # Verificar que la rotación sea válida
            if not np.allclose(np.dot(rotation_matrix, rotation_matrix.T), np.eye(3), atol=1e-3):
                self.get_logger().warn("Invalid rotation matrix")
                return
                
            q = R.from_matrix(rotation_matrix).as_quat()
            
            t.transform.translation.x = translation[0]
            t.transform.translation.y = translation[1]
            t.transform.translation.z = translation[2]
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            
            self.tf_broadcaster.sendTransform(t)
        except Exception as e:
            self.get_logger().error(f"Error publishing transform: {e}")

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
    
    def stop_robot(self):
        """Detener el robot enviando velocidades cero"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)
    
    def inflate_map(self):
        """Inflar obstáculos en el mapa para navegación segura"""
        from scipy import ndimage
        
        # Crear mapa binario: 1 = obstáculo, 0 = libre
        obstacle_map = (self.grid > 50).astype(np.uint8)
        
        # Crear kernel de dilatación
        kernel_size = 2 * self.safety_margin_cells + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Dilatar obstáculos
        inflated = ndimage.binary_dilation(obstacle_map, structure=kernel)
        
        # Crear grid inflado: -1 = desconocido, 0 = libre, 100 = obstáculo
        self.inflated_grid = np.full_like(self.grid, -1, dtype=np.int8)
        self.inflated_grid[self.grid == 0] = 0  # Libre
        self.inflated_grid[inflated] = 100  # Obstáculo inflado
    

    def publish_path(self, path_msg):
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.map_frame_id
        self.path_pub.publish(path_msg)

    def a_star_planning(self, start_pos, goal_pos):
        """
        Algoritmo A* para planificación de rutas
        Args:
            start_pos: (x, y) posición inicial en coordenadas del mundo
            goal_pos: (x, y) posición objetivo en coordenadas del mundo
        Returns:
            Lista de poses (x, y) que forman el camino, o None si no se encuentra
        """
        if self.inflated_grid is None:
            self.get_logger().error("Inflated grid not available for path planning")
            return None
            
        # Convertir a coordenadas de grid
        start_gx, start_gy = self.world_to_grid(start_pos[0], start_pos[1])
        goal_gx, goal_gy = self.world_to_grid(goal_pos[0], goal_pos[1])
        
        # Verificar que start y goal estén dentro del mapa
        h, w = self.inflated_grid.shape
        if not (0 <= start_gx < w and 0 <= start_gy < h):
            self.get_logger().error(f"Start position ({start_gx}, {start_gy}) is outside map bounds")
            return None
        if not (0 <= goal_gx < w and 0 <= goal_gy < h):
            self.get_logger().error(f"Goal position ({goal_gx}, {goal_gy}) is outside map bounds")
            return None
            
        # Verificar que start y goal estén en espacio libre
        if self.inflated_grid[start_gy, start_gx] != 0:
            self.get_logger().error("Start position is not in free space")
            return None
        if self.inflated_grid[goal_gy, goal_gx] != 0:
            self.get_logger().error("Goal position is not in free space")
            return None
            
        # Inicializar estructuras de A*
        open_set = []
        heapq.heappush(open_set, (0, start_gx, start_gy))
        came_from = {}
        g_score = {(start_gx, start_gy): 0}
        f_score = {(start_gx, start_gy): self.heuristic((start_gx, start_gy), (goal_gx, goal_gy))}
        
        # Direcciones de movimiento (8-conectividad)
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        while open_set:
            # Obtener nodo con menor f_score
            current_f, current_x, current_y = heapq.heappop(open_set)
            current = (current_x, current_y)
            
            # Si llegamos al objetivo
            if current == (goal_gx, goal_gy):
                # Reconstruir camino
                path = []
                while current in came_from:
                    gx, gy = current
                    wx, wy = self.grid_to_world(gx, gy)
                    path.append((wx, wy))
                    current = came_from[current]
                
                # Agregar punto inicial
                wx, wy = self.grid_to_world(start_gx, start_gy)
                path.append((wx, wy))
                path.reverse()
                
                return path
                
            # Explorar vecinos
            for dx, dy in directions:
                neighbor_x = current_x + dx
                neighbor_y = current_y + dy
                neighbor = (neighbor_x, neighbor_y)
                
                # Verificar límites
                if not (0 <= neighbor_x < w and 0 <= neighbor_y < h):
                    continue
                    
                # Verificar obstáculos
                if self.inflated_grid[neighbor_y, neighbor_x] != 0:
                    continue
                    
                # Calcular costo del movimiento
                move_cost = np.sqrt(dx**2 + dy**2)  # Distancia euclidiana
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, (goal_gx, goal_gy))
                    heapq.heappush(open_set, (f_score[neighbor], neighbor_x, neighbor_y))
                    
        # No se encontró camino
        return None
    
    def heuristic(self, pos1, pos2):
        """Función heurística para A* (distancia euclidiana)"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def smooth_path(self, path):
        """Suavizar el camino eliminando puntos innecesarios"""
        if len(path) <= 2:
            return path
            
        smoothed_path = [path[0]]  # Empezar con el primer punto
        
        i = 0
        while i < len(path) - 1:
            j = i + 1
            # Encontrar el punto más lejano al que se puede ir en línea recta
            while j < len(path):
                if self.is_line_free(path[i], path[j]):
                    j += 1
                else:
                    break
            
            # Agregar el último punto válido
            smoothed_path.append(path[j - 1])
            i = j - 1
            
        # Asegurar que el último punto esté incluido
        if smoothed_path[-1] != path[-1]:
            smoothed_path.append(path[-1])
            
        return smoothed_path
    
    def densify_path(self, path, spacing=0.01, max_points=1000):
        """
        Interpola puntos entre los puntos de la ruta para aumentar la densidad, evitando bucles infinitos y rutas excesivas.
        Args:
            path: lista de tuplas (x, y)
            spacing: distancia deseada entre puntos (metros)
            max_points: máximo de puntos permitidos en la ruta densificada
        Returns:
            Lista densificada de puntos (x, y)
        """
        if len(path) < 2:
            return path
        dense_path = [path[0]]
        total_points = 1
        for i in range(1, len(path)):
            p0 = np.array(path[i-1])
            p1 = np.array(path[i])
            segment = p1 - p0
            length = np.linalg.norm(segment)
            if length < 1e-5:
                continue  # Ignorar segmentos muy cortos
            n_points = int(np.floor(length / spacing))
            for j in range(1, n_points+1):
                if total_points >= max_points:
                    break
                new_point = p0 + (segment * (j * spacing / length))
                if not np.allclose(new_point, dense_path[-1], atol=1e-6):
                    dense_path.append(tuple(new_point))
                    total_points += 1
            if total_points >= max_points:
                break
        # Asegurar que el último punto esté incluido
        if not np.allclose(dense_path[-1], path[-1], atol=1e-6):
            dense_path.append(path[-1])
        return dense_path

    def is_line_free(self, start, end):
        """Verificar si la línea entre dos puntos está libre de obstáculos"""
        # Usar el algoritmo de Bresenham en el grid
        gx1, gy1 = self.world_to_grid(start[0], start[1])
        gx2, gy2 = self.world_to_grid(end[0], end[1])
        
        points = self.bresenham_line(gx1, gy1, gx2, gy2)
        
        h, w = self.inflated_grid.shape
        for gx, gy in points:
            if not (0 <= gx < w and 0 <= gy < h):
                return False
            if self.inflated_grid[gy, gx] != 0:
                return False
                
        return True
    
    def bresenham_line(self, x1, y1, x2, y2):
        """Algoritmo de Bresenham para obtener puntos en una línea"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while True:
            points.append((x, y))
            
            if x == x2 and y == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
                
        return points

    def pure_pursuit_control(self, current_pose, path):
        """
        Controlador Pure Pursuit adaptado del ejemplo de Atsushi Sakai.
        Args:
            current_pose: La pose actual del robot (Pose)
            path: La ruta a seguir (lista de tuplas (x, y))
        Returns:
            Comando de velocidad (Twist)
        """
        import math
        if not path or len(path) < 2:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd

        # Obtener posición y orientación actual
        robot_x = current_pose.position.x
        robot_y = current_pose.position.y
        q = current_pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # Buscar el punto más cercano en el path
        dists = [math.hypot(robot_x - px, robot_y - py) for px, py in path]
        nearest_idx = int(np.argmin(dists))

        # Buscar el índice del lookahead point
        lookahead = self.lookahead_distance
        ind = nearest_idx
        while ind < len(path) - 1 and math.hypot(path[ind][0] - robot_x, path[ind][1] - robot_y) < lookahead:
            ind += 1
        target_x, target_y = path[ind]

        # Calcular ángulo al objetivo
        dx = target_x - robot_x
        dy = target_y - robot_y
        alpha = math.atan2(dy, dx) - yaw
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi  # Normalizar [-pi, pi]

        # Calcular curvatura y velocidad angular
        Lf = lookahead
        curvature = 2.0 * math.sin(alpha) / Lf if Lf > 1e-6 else 0.0
        v = self.linear_velocity
        omega = v * curvature

        # Limitar velocidad angular para suavidad
        max_angular_vel = 0.25
        omega = np.clip(omega, -max_angular_vel, max_angular_vel)

        # Reducir velocidad lineal si hay mucho giro
        if abs(omega) > 0.2:
            v = self.linear_velocity * 0.8

        # Conservadurismo moderado: reducir velocidad si hay obstáculos muy cercanos
        if self.latest_scan is not None:
            ranges = np.array(self.latest_scan.ranges)
            valid_ranges = ranges[np.isfinite(ranges) & (ranges > 0)]
            if len(valid_ranges) > 0:
                min_distance = min(valid_ranges)
                if min_distance < 0.5:
                    safety_factor = max(0.6, min_distance / 0.8)
                    v *= safety_factor

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        return cmd

    def check_for_obstacles(self):
        """
        Verifica si hay obstáculos en un campo de visión frontal más amplio (±90°).
        """
        if self.latest_scan is None:
            return False

        # Ampliar el campo de visión frontal a ±90°
        front_angle_range = np.deg2rad(120)  # 180 grados total
        center_index = len(self.latest_scan.ranges) // 2
        angle_increment = self.latest_scan.angle_increment
        if angle_increment == 0.0:
            return False # Evitar división por cero si el scan no es válido

        num_indices_side = int(front_angle_range / 2 / angle_increment)
        start_index = max(0, center_index - num_indices_side)
        end_index = min(len(self.latest_scan.ranges), center_index + num_indices_side)

        # Extraer los rangos frontales
        front_ranges = self.latest_scan.ranges[start_index:end_index]

        # Filtrar rangos inválidos (inf, nan)
        valid_front_ranges = [r for r in front_ranges if np.isfinite(r) and r > 0]

        if not valid_front_ranges:
            return False

        # Reducir la distancia de detección a 0.15 m
        obstacle_distance = 0.2
        self.obstacle_detection_distance = obstacle_distance  # Actualizar el parámetro en el nodo

        min_front_distance = min(valid_front_ranges)
        if min_front_distance < obstacle_distance:
            self.get_logger().warn(f"Obstacle detected at {min_front_distance:.2f}m (threshold: {obstacle_distance:.2f}m).")
            return True

        return False

    def get_obstacle_avoidance_direction(self):
        """
        Evalúa el promedio de distancia a la izquierda y derecha del campo de visión frontal
        y retorna 1 si conviene girar a la izquierda, -1 si conviene girar a la derecha.
        """
        if self.latest_scan is None:
            return -1  # Por defecto derecha

        front_angle_range = np.deg2rad(60)
        center_index = len(self.latest_scan.ranges) // 2
        angle_increment = self.latest_scan.angle_increment
        if angle_increment == 0.0:
            return -1
        num_indices_side = int(front_angle_range / 2 / angle_increment)
        start_index = max(0, center_index - num_indices_side)
        end_index = min(len(self.latest_scan.ranges), center_index + num_indices_side)
        front_ranges = self.latest_scan.ranges[start_index:end_index]
        valid_front_ranges = [r for r in front_ranges if np.isfinite(r) and r > 0]
        if not valid_front_ranges:
            return -1
        # Dividir en izquierda y derecha
        n = len(front_ranges)
        left = [r for r in front_ranges[n//2:] if np.isfinite(r) and r > 0]
        right = [r for r in front_ranges[:n//2] if np.isfinite(r) and r > 0]
        avg_left = np.mean(left) if left else 0
        avg_right = np.mean(right) if right else 0
        if avg_left > avg_right:
            return 1  # Izquierda
        else:
            return -1  # Derecha


def main(args=None):
    rclpy.init(args=args)
    node = AmclNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()