import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float64
from sensor_msgs.msg import Joy
from st3215 import ST3215
import time

class ServoController(Node):
    def __init__(self):
        super().__init__('actuators_pan')
        self.subscription = self.create_subscription(
            Int32,
            '/servo_command_pan',
            self.servo_callback,
            10
        )
        self.position_publisher = self.create_publisher(
            Float64,
            '/servo_position_pan',
            10
        )
        self.timer = self.create_timer(0.1, self.timer_callback)

    servo = ST3215('/dev/ttyACM0')

    def servo_callback(self, msg):
        command = msg.data
        if command == 1:
            self.servo.StartServo(1)
            self.servo.MoveTo(1, int((300 + 90) * (4095 / 360)), 500)  # Move +90 from home
        elif command == -1:
            self.servo.StartServo(1)
            self.servo.MoveTo(1, int((300 - 90) * (4095 / 360)), 500)  # Move -90 from home
        elif command == 2:
            self.servo.StartServo(1)
            self.servo.MoveTo(1, int(300 * (4095 / 360)), 500)  # Move to home (300 deg)
        else:
            self.servo.StopServo(1)

    def timer_callback(self):
        pos = self.servo.ReadPosition(1)
        if pos is not None:
            msg = Float64()
            msg.data = float(pos * (360.0 / 4095.0))
            self.position_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ServoController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()