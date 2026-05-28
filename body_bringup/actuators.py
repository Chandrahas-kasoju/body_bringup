#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from rcl_interfaces.msg import SetParametersResult
from st3215 import ST3215
import time

class PIDController:
    """A standard generic PID Controller implementation"""
    def __init__(self, kp, ki, kd, output_limits=(None, None), deadband=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.min_out, self.max_out = output_limits
        self.deadband = deadband

    def compute(self, setpoint, measured_value, dt):
        raw_error = setpoint - measured_value
        
        # Apply deadband to error
        if abs(raw_error) < self.deadband:
            error = 0.0
        else:
            error = raw_error
            
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        
        # Anti-windup / Output clamping
        if self.min_out is not None:
            output = max(self.min_out, output)
        if self.max_out is not None:
            output = min(self.max_out, output)
            
        return output

class GenericServoController(Node):
    def __init__(self):
        super().__init__('generic_servo_controller')
        
        # Declare parameters for flexibility
        self.declare_parameter('kp', 110.0)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 5.0)
        self.declare_parameter('servo_port', '/dev/ttyACM0')
        self.declare_parameter('servo_id', 1)
        self.declare_parameter('use_software_pid', True) # Set to True to use custom PID
        self.declare_parameter('max_accel', 1000.0) # Max change in speed (steps/s) per second
        self.declare_parameter('max_decel', 1000.0) # Allow faster braking to prevent overshoot
        
        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        port = self.get_parameter('servo_port').value
        self.sts_id = self.get_parameter('servo_id').value
        self.use_software_pid = self.get_parameter('use_software_pid').value
        self.max_accel = self.get_parameter('max_accel').value
        self.max_decel = self.get_parameter('max_decel').value
        
        self.current_speed_cmd = 0.0
        
        self.get_logger().info(f"Connecting to servo on {port}")
        self.servo = ST3215(port)
        
        if not self.servo.PingServo(self.sts_id):
            self.get_logger().error(f"Servo {self.sts_id} not connected!")
            
        # Target angle in degrees
        self.target_angle_deg = 300.0  # Default starting target
        
        # Initialize PID controller for calculating speed based on position error
        # Max speed for ST3215 is roughly 3400 steps/s. Increased output limits so it can fight gravity.
        # Added a 0.5 degree deadband to provide tolerance and stop jittering once it reaches the target.
        self.pid = PIDController(self.kp, self.ki, self.kd, output_limits=(-400, 400), deadband=1.0)
        
        # Register parameter callback for dynamic tuning
        self.add_on_set_parameters_callback(self.parameters_callback)
        
        # Generic ROS approach: subscribe to a Float64 for the target angle
        self.subscription = self.create_subscription(
            Float64,
            '/servo_command_pan',
            self.target_callback,
            10
        )
        
        # Publishers for PID tuning with rqt_plot
        self.current_angle_pub = self.create_publisher(Float64, '/servo_current_angle_pan', 10)
        self.target_angle_pub = self.create_publisher(Float64, '/servo_target_angle_pan', 10)
        
        # Control loop timer (e.g., 20 Hz)
        self.timer_period = 0.05
        self.timer = self.create_timer(self.timer_period, self.control_loop)
        self.last_time = time.time()
        
        
    def parameters_callback(self, params):
        for param in params:
            if param.name == 'kp':
                self.kp = param.value
                self.pid.kp = self.kp
                self.get_logger().info(f"Updated kp to {self.kp}")
            elif param.name == 'ki':
                self.ki = param.value
                self.pid.ki = self.ki
                self.get_logger().info(f"Updated ki to {self.ki}")
            elif param.name == 'kd':
                self.kd = param.value
                self.pid.kd = self.kd
                self.get_logger().info(f"Updated kd to {self.kd}")
            elif param.name == 'max_accel':
                self.max_accel = param.value
                self.get_logger().info(f"Updated max_accel to {self.max_accel}")
            elif param.name == 'max_decel':
                self.max_decel = param.value
                self.get_logger().info(f"Updated max_decel to {self.max_decel}")
        return SetParametersResult(successful=True)

    def target_callback(self, msg):
        """Callback to update the target angle from ROS topic"""
        self.target_angle_deg = msg.data
        self.get_logger().info(f"Received new target angle: {self.target_angle_deg} degrees")
        
    def control_loop(self):
        """Main control loop running at fixed frequency"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if self.use_software_pid:
            # === OPTION A: SOFTWARE VELOCITY PID ===
            # Read current position in steps
            current_pos_steps = self.servo.ReadPosition(self.sts_id)
            if current_pos_steps is None:
                return
                
            # Convert step position to angle (0-4095 steps = 0-360 degrees)
            current_angle_deg = (current_pos_steps / 4095.0) * 360.0
            
            # Publish for tuning/plotting
            self.current_angle_pub.publish(Float64(data=current_angle_deg))
            self.target_angle_pub.publish(Float64(data=self.target_angle_deg))
            
            # Compute PID control signal (desired velocity)
            raw_speed_cmd = self.pid.compute(self.target_angle_deg, current_angle_deg, dt)
            
            # Determine if we are accelerating or decelerating
            # If the raw command is pulling the speed closer to 0 (or opposite direction), we are braking
            is_decelerating = False
            if (self.current_speed_cmd > 0 and raw_speed_cmd < self.current_speed_cmd) or \
               (self.current_speed_cmd < 0 and raw_speed_cmd > self.current_speed_cmd):
                is_decelerating = True
                
            # Apply acceleration limit (slew rate limiting)
            allowed_delta = (self.max_decel * dt) if is_decelerating else (self.max_accel * dt)
            
            if raw_speed_cmd > self.current_speed_cmd + allowed_delta:
                self.current_speed_cmd += allowed_delta
            elif raw_speed_cmd < self.current_speed_cmd - allowed_delta:
                self.current_speed_cmd -= allowed_delta
            else:
                self.current_speed_cmd = raw_speed_cmd
            
            # Use Rotate() to set the speed. The sign dictates direction.
            # CRITICAL FIX: We must NOT use StopServo() here when error is small. 
            # If we stop the servo, it loses holding torque and gravity immediately pulls it down, 
            # causing endless bouncing. The PID integral term will naturally find the exact 
            # speed command needed to hold the weight still.
            self.servo.Rotate(self.sts_id, int(self.current_speed_cmd))
                
        else:
            # === OPTION B: HARDWARE POSITION CONTROLLER (STANDARD) ===
            # ST3215 already has an internal hardware PID loop optimized for the motor.
            # Usually, standard robotics systems just send the mapped target position directly.
            target_steps = int((self.target_angle_deg / 360.0) * 4095.0)
            
            # Clamp to safe physical limits (e.g., 0 to 4095)
            target_steps = max(0, min(4095, target_steps))
            
            # Send position to the servo's internal controller
            self.servo.MoveTo(self.sts_id, target_steps, 500, 50) 

def main(args=None):
    rclpy.init(args=args)
    node = GenericServoController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop servo before shutting down
        node.servo.StopServo(node.sts_id)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
