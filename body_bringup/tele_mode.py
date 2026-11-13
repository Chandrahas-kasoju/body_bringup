#!/usr/bin/env python3
"""Owns Tele's current mode and runs the Christmas show.

The mode is published on /tele_mode once a second, so every node on the head
and the body can follow it. It is changed either with a message on
/tele_mode_request or through a small local web address that the touch-screen
GUI uses (the GUI runs outside ROS):

    GET  http://127.0.0.1:8011/mode            -> {"mode": "PATROL"}
    POST http://127.0.0.1:8011/mode/CHRISTMAS  -> switches mode
    POST http://127.0.0.1:8011/about           -> Tele introduces itself out loud

PATROL is the normal working mode. In CHRISTMAS mode this node plays Jingle Bells on the speaker and sways the
head slowly. The LED node and the head's face tracker read /tele_mode themselves.
"""
import json
import math
import os
import subprocess
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String

MODES = ('PATROL', 'CHRISTMAS')
HTTP_PORT = 8011
SONG_FILE = '/tmp/tele_jingle_bells.wav'

# "About Me": what Tele says about itself, per language. A language is only used
# when its Piper voice file is installed; otherwise Tele falls back to English.
ABOUT_FILE = '/tmp/tele_about.wav'
# Speech settings. Stretching every sound (a high length scale) is what makes a
# voice robotic, so the pace stays close to natural and clarity comes from pauses
# between sentences. The two noise values add the small variations in tone and
# rhythm that a human voice has; this voice's own defaults (0.5 and 0.33) are flat.
ABOUT_VOICE_ARGS = ['--length-scale', '1.1', '--sentence-silence', '0.55',
                    '--noise-scale', '0.667', '--noise-w-scale', '0.8']
VOICES = {
    'en': '/root/piper-tts/en_GB-jenny_dioco-medium.onnx',
    'de': '/root/piper-tts/de_DE-thorsten-medium.onnx',
}
ABOUT_TEXT = {
    'en': "Hi, I'm Tele. I'm a patrol robot for hospitals. "
          "I find my way to patients with a thermal camera and a radar. "
          "That way, I can look after them, and their privacy stays protected.",
    'de': 'Hallo, ich bin Tele. Ich bin ein Patrouillenroboter für Krankenhäuser. '
          'Patientinnen und Patienten erkenne ich mit einer Wärmebildkamera und einem Radar. '
          'So kann ich auf sie achten, und ihre Privatsphäre bleibt geschützt.',
}

# Head pose: pan is driven by the body, tilt by the head
PAN_HOME, TILT_HOME = 300.0, 50.0
PAN_SWAY_DEG, TILT_SWAY_DEG = 15.0, 4.0
TILT_SWAY_CENTRE = 55.0
SWAY_PERIOD_S = 10.0  # One full left-right swing; keeps the head below 10 deg/s

# Jingle Bells (James Pierpont, 1857; public domain), as (note, beats)
VERSE = [
    ('G4', 1), ('E5', 1), ('D5', 1), ('C5', 1), ('G4', 3), ('G4', .5), ('G4', .5),
    ('G4', 1), ('E5', 1), ('D5', 1), ('C5', 1), ('A4', 4),
    ('A4', 1), ('F5', 1), ('E5', 1), ('D5', 1), ('B4', 4),
    ('G5', 1), ('G5', 1), ('F5', 1), ('D5', 1), ('E5', 4),
    ('G4', 1), ('E5', 1), ('D5', 1), ('C5', 1), ('G4', 4),
    ('G4', 1), ('E5', 1), ('D5', 1), ('C5', 1), ('A4', 3), ('A4', 1),
    ('A4', 1), ('F5', 1), ('E5', 1), ('D5', 1), ('G5', 1), ('G5', 1), ('G5', 1), ('G5', 1),
    ('A5', 1), ('G5', 1), ('F5', 1), ('D5', 1), ('C5', 2), ('G5', 2),
]
CHORUS_START = [
    ('E5', 1), ('E5', 1), ('E5', 2), ('E5', 1), ('E5', 1), ('E5', 2),
    ('E5', 1), ('G5', 1), ('C5', 1.5), ('D5', .5), ('E5', 4),
    ('F5', 1), ('F5', 1), ('F5', 1.5), ('F5', .5), ('F5', 1), ('E5', 1), ('E5', 1), ('E5', .5), ('E5', .5),
]
CHORUS = (CHORUS_START + [('E5', 1), ('D5', 1), ('D5', 1), ('E5', 1), ('D5', 2), ('G5', 2)]
          + CHORUS_START + [('G5', 1), ('G5', 1), ('F5', 1), ('D5', 1), ('C5', 4)])
SEMITONES = {'C': -9, 'D': -7, 'E': -5, 'F': -4, 'G': -2, 'A': 0, 'B': 2}


def note_hz(name):
    return 440.0 * 2 ** ((SEMITONES[name[0]] + 12 * (int(name[1]) - 4)) / 12)


def write_song(path, beat_s=0.3, rate=22050):
    """Synthesise the tune with a bell-like tone and save it as a WAV file."""
    notes = VERSE + CHORUS
    total = sum(beats for _, beats in notes) * beat_s + 1.5
    audio = np.zeros(int(total * rate))
    start = 0.0
    for name, beats in notes:
        t = np.arange(int(min(beats * beat_s + 0.8, 2.0) * rate)) / rate
        f = note_hz(name)
        tone = (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2 * f * t)
                + 0.25 * np.sin(2 * np.pi * 3 * f * t) + 0.12 * np.sin(2 * np.pi * 4.2 * f * t))
        tone *= np.exp(-3.5 * t) * np.minimum(t / 0.005, 1.0)  # Soft attack, ringing decay
        i = int(start * rate)
        audio[i:i + len(tone)] += tone[:len(audio) - i]
        start += beats * beat_s
    audio = 0.6 * audio / np.max(np.abs(audio))
    with wave.open(path, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes((audio * 32767).astype(np.int16).tobytes())


class TeleMode(Node):

    def __init__(self):
        super().__init__('tele_mode')
        self.mode = 'PATROL'
        self.lock = threading.Lock()
        self.mode_pub = self.create_publisher(String, '/tele_mode', 10)
        self.pan_pub = self.create_publisher(Float64, '/servo_command_pan', 10)
        self.tilt_pub = self.create_publisher(Float64, '/servo_command', 10)
        self.create_subscription(String, '/tele_mode_request', lambda m: self.set_mode(m.data), 10)
        self.create_timer(1.0, self.publish_mode)
        self.create_timer(0.2, self.sway)
        self.declare_parameter('about_language', 'en')
        self.about_seconds = 0.0
        self.about_player = None
        self.show_started = 0.0
        self.home_until = 0.0
        self.player = None
        try:
            write_song(SONG_FILE)
        except Exception as e:
            self.get_logger().error(f'Could not create the song: {e}')
        threading.Thread(target=self.music_loop, daemon=True).start()
        threading.Thread(target=self.prepare_about, daemon=True).start()
        threading.Thread(target=self.serve_http, daemon=True).start()
        self.get_logger().info(f'Mode manager started in {self.mode} mode, web address on port {HTTP_PORT}')

    # ---- mode handling
    def set_mode(self, requested):
        requested = requested.strip().upper()
        if requested not in MODES:
            self.get_logger().warn(f'Unknown mode requested: {requested}')
            return False
        with self.lock:
            if requested == self.mode:
                return True
            previous, self.mode = self.mode, requested
        self.get_logger().info(f'Mode changed: {previous} -> {requested}')
        if requested == 'CHRISTMAS':
            self.show_started = time.time()
        if previous == 'CHRISTMAS':
            self.stop_music()
            # The sway timer sends the head home for the next two seconds. A single
            # message from here could be overtaken by a last sway command.
            self.home_until = time.time() + 2.0
        self.publish_mode()
        return True

    def publish_mode(self):
        self.mode_pub.publish(String(data=self.mode))

    # ---- Christmas show
    def sway(self):
        if self.mode != 'CHRISTMAS':
            if time.time() < self.home_until:
                self.pan_pub.publish(Float64(data=PAN_HOME))
                self.tilt_pub.publish(Float64(data=TILT_HOME))
            return
        phase = 2 * math.pi * (time.time() - self.show_started) / SWAY_PERIOD_S
        self.pan_pub.publish(Float64(data=PAN_HOME + PAN_SWAY_DEG * math.sin(phase)))
        # A small nod, twice per swing
        self.tilt_pub.publish(Float64(data=TILT_SWAY_CENTRE + TILT_SWAY_DEG * math.sin(2 * phase)))

    def music_loop(self):
        while rclpy.ok():
            if self.mode == 'CHRISTMAS' and os.path.exists(SONG_FILE):
                try:
                    self.player = subprocess.Popen(['aplay', '-q', SONG_FILE])
                    self.player.wait()
                except Exception as e:
                    self.get_logger().error(f'Could not play the song: {e}')
                    time.sleep(5)
                time.sleep(0.5)  # Short pause before the song repeats
            else:
                time.sleep(0.2)

    def stop_music(self):
        player = self.player
        if player is not None and player.poll() is None:
            player.terminate()

    # ---- About Me
    def prepare_about(self):
        """Speak the introduction into a file once, so pressing the button answers at once."""
        language = self.get_parameter('about_language').value
        if language not in VOICES or not os.path.exists(VOICES[language]):
            if language != 'en':
                self.get_logger().warn(f'No voice installed for "{language}", using English')
            language = 'en'
        try:
            subprocess.run(
                ['piper', '--model', VOICES[language], *ABOUT_VOICE_ARGS, '--output-file', ABOUT_FILE],
                input=ABOUT_TEXT[language].encode('utf-8'), capture_output=True, timeout=120, check=True)
            with wave.open(ABOUT_FILE) as f:
                self.about_seconds = f.getnframes() / f.getframerate()
            self.get_logger().info(f'About Me speech ready ({language}, {self.about_seconds:.0f} s)')
        except Exception as e:
            self.get_logger().error(f'Could not prepare the About Me speech: {e}')

    def speak_about(self):
        """Returns how many seconds the speech takes, or 0 if it cannot be spoken now."""
        if self.about_seconds <= 0:
            return 0.0
        if self.about_player is not None and self.about_player.poll() is None:
            return self.about_seconds  # Already speaking
        if self.mode == 'CHRISTMAS':
            self.set_mode('PATROL')  # Not over the music
            time.sleep(0.3)
        self.about_player = subprocess.Popen(['aplay', '-q', ABOUT_FILE])
        self.get_logger().info('Speaking the About Me introduction')
        return self.about_seconds

    # ---- web address for the GUI
    def serve_http(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def reply(self, code, data):
                body = json.dumps(data).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.rstrip('/') == '/mode':
                    self.reply(200, {'mode': node.mode, 'modes': list(MODES)})
                else:
                    self.reply(404, {'error': 'not found'})

            def do_POST(self):
                parts = self.path.strip('/').split('/')
                if parts == ['about']:
                    seconds = node.speak_about()
                    self.reply(200 if seconds else 503, {'speaking': seconds > 0, 'seconds': seconds})
                    return
                if len(parts) == 2 and parts[0] == 'mode' and node.set_mode(parts[1]):
                    self.reply(200, {'mode': node.mode})
                else:
                    self.reply(400, {'error': 'unknown mode', 'modes': list(MODES)})

            def log_message(self, *args):
                pass

        ThreadingHTTPServer.allow_reuse_address = True
        ThreadingHTTPServer(('127.0.0.1', HTTP_PORT), Handler).serve_forever()


def main(args=None):
    rclpy.init(args=args)
    node = TeleMode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_music()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
