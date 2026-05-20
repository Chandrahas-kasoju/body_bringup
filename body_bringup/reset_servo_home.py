#!/usr/bin/env python3
import sys
import time
import argparse
from st3215 import ST3215

def main():
    parser = argparse.ArgumentParser(description='Reset or set the home position of an ST3215 servo.')
    parser.add_argument('--port', type=str, default='/dev/ttyACM0', help='Serial port (default: /dev/ttyACM0)')
    parser.add_argument('--id', type=int, default=1, help='Servo ID (default: 1)')
    parser.add_argument('--clear', action='store_true', help='Clear the existing offset (reset to factory home)')
    parser.add_argument('--set-current', action='store_true', help='Set the current physical position as the new home (0 position)')
    
    args = parser.parse_args()

    try:
        servo = ST3215(args.port)
    except Exception as e:
        print(f"Failed to connect to servo on {args.port}: {e}")
        return

    servo_id = args.id

    if not servo.PingServo(servo_id):
        print(f"Could not find servo with ID {servo_id} on {args.port}.")
        return

    print(f"Found servo ID {servo_id}.")

    # Unlock EEPROM to allow modifications
    print("Unlocking EEPROM...")
    servo.UnLockEprom(servo_id)
    time.sleep(0.1)

    if args.clear:
        print("Clearing position offset...")
        servo.CorrectPosition(servo_id, 0)
        time.sleep(0.1)
        new_pos = servo.ReadPosition(servo_id)
        print(f"Offset cleared. Current position is now: {new_pos}")
        
    elif args.set_current:
        # First clear offset to get the true raw position
        print("Temporarily clearing offset to read raw physical position...")
        servo.CorrectPosition(servo_id, 0)
        time.sleep(0.5) # Wait for changes to take effect
        
        raw_pos = servo.ReadPosition(servo_id)
        print(f"Raw physical position: {raw_pos}")
        
        # Calculate the correction required
        # ST3215 correction is typically a signed 12-bit value (-2048 to +2047)
        correction = raw_pos
        if correction > 2047:
            correction = correction - 4096
            
        print(f"Applying correction offset of {correction}...")
        servo.CorrectPosition(servo_id, correction)
        time.sleep(0.5) # Wait for changes to take effect
        
        new_pos = servo.ReadPosition(servo_id)
        print(f"New position is now: {new_pos} (Target: 0)")
        
    else:
        print("\nNo action specified. Use --clear or --set-current")
        print(f"Current position: {servo.ReadPosition(servo_id)}")
        print(f"Current correction: {servo.ReadCorrection(servo_id)}")

    # Always lock EEPROM after modifications to prevent accidental corruption
    print("Locking EEPROM...")
    servo.LockEprom(servo_id)
    print("Done.")

if __name__ == '__main__':
    main()
