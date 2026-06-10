import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import subprocess
import threading
import tempfile
import os
import time

# --- Configuration ---
PIPER_EXECUTABLE  = "piper"
# Updated to use a standard Danish Piper TTS model
PIPER_MODEL_PATH  = "/home/manu/da_DK-talesyntese-medium.onnx" 
# Translated to Danish
SPEECH_TEXT       = "Hej, jeg er Tely. Tryk venligst på knappen på skærmen for at få hjælp." 
SPEECH_SPEED      = 1.2
REPEAT_COOLDOWN_S = 10.0   # minimum seconds between any two greetings


class RobotSpeechNode(Node):

    def __init__(self):
        super().__init__('robot_speech_node')

        self.subscription = self.create_subscription(
            String,
            '/person_intent',
            self.intent_callback,
            10
        )
        self.get_logger().info("Robot Speech Node started. Listening to /person_intent")

        self._lock             = threading.Lock()
        self._is_speaking      = False
        self._has_greeted      = False
        self._last_spoken_time = 0.0   # epoch timestamp of the last greeting


    # ------------------------------------------------------------------
    # ROS 2 Callback
    # ------------------------------------------------------------------

    def intent_callback(self, msg: String):
        intent = msg.data.strip().upper()

        if intent == 'WANT_TO_INTERACT':
            with self._lock:
                now             = time.time()
                time_since_last = now - self._last_spoken_time
                cooldown_ok     = time_since_last >= REPEAT_COOLDOWN_S

                if not self._is_speaking and not self._has_greeted and cooldown_ok:
                    # All three gates pass → speak
                    self._is_speaking      = True
                    self._has_greeted      = True
                    self._last_spoken_time = now
                    thread = threading.Thread(target=self._speak_async, daemon=True)
                    thread.start()
                    self.get_logger().info("WANT_TO_INTERACT → speaking greeting")

                elif not cooldown_ok:
                    # Cooldown still active — log remaining time
                    remaining = REPEAT_COOLDOWN_S - time_since_last
                    self.get_logger().info(
                        f"Cooldown active — {remaining:.1f}s left before next greeting allowed"
                    )

                elif self._is_speaking:
                    self.get_logger().debug("Already speaking — ignored")

                elif self._has_greeted:
                    self.get_logger().debug("Already greeted this encounter — ignored")

        else:
            # Any other intent resets the per-encounter flag
            # but the cooldown timer (_last_spoken_time) is NOT reset
            with self._lock:
                self._has_greeted = False
            self.get_logger().debug(f"Intent '{intent}' → _has_greeted reset")


    # ------------------------------------------------------------------
    # TTS Worker (background thread)
    # ------------------------------------------------------------------

    def _speak_async(self):
        tmp_wav = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                tmp_wav = f.name

            piper_result = subprocess.run(
                [
                    PIPER_EXECUTABLE,
                    '--model',        PIPER_MODEL_PATH,
                    '--length-scale', str(SPEECH_SPEED),
                    '--output-file',  tmp_wav,
                ],
                input=SPEECH_TEXT.encode('utf-8'),
                capture_output=True,
                timeout=10,
            )

            if piper_result.returncode != 0:
                self.get_logger().error(
                    f"Piper failed: {piper_result.stderr.decode().strip()}"
                )
                return

            subprocess.run(['aplay', tmp_wav], timeout=15)

        except FileNotFoundError as e:
            self.get_logger().error(f"Executable not found: {e}")
        except subprocess.TimeoutExpired:
            self.get_logger().error("Speech timed out.")
        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                os.remove(tmp_wav)
            with self._lock:
                self._is_speaking = False


# ----------------------------------------------------------------------
# Entry Point
# ----------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = RobotSpeechNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.context.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
