#!/usr/bin/env python3
"""
AI Trade Bot - Stop Analysis
============================

Stops the running analysis gracefully.

Usage:
    python stop_analysis.py
"""

import os
import sys
import signal
import time
from pathlib import Path

PID_FILE = Path(__file__).parent / "data" / ".analysis.pid"


def main():
    """Stop the running analysis."""
    print("\n🛑 Stopping AI Trade Bot...")

    if not PID_FILE.exists():
        print("   No running analysis found.")
        print("   (PID file doesn't exist)")
        sys.exit(0)

    # Read PID
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, FileNotFoundError):
        print("   Invalid PID file. Cleaning up...")
        PID_FILE.unlink()
        sys.exit(0)

    # Check if process exists
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        print(f"   Process {pid} not running. Cleaning up...")
        PID_FILE.unlink()
        sys.exit(0)

    # Send SIGTERM
    print(f"   Sending stop signal to process {pid}...")
    try:
        os.kill(pid, signal.SIGTERM)
    except PermissionError:
        print("   ⚠️  Permission denied. Try: sudo python stop_analysis.py")
        sys.exit(1)

    # Wait for process to stop
    for i in range(30):
        try:
            os.kill(pid, 0)
            print(f"   Waiting for shutdown... ({i+1}/30)")
            time.sleep(1)
        except ProcessLookupError:
            break

    # Verify stopped
    try:
        os.kill(pid, 0)
        print("   ⚠️  Process didn't stop. Force killing...")
        os.kill(pid, signal.SIGKILL)
        time.sleep(1)
    except ProcessLookupError:
        pass

    # Clean up PID file
    if PID_FILE.exists():
        PID_FILE.unlink()

    print("\n✅ Analysis stopped successfully!")
    print("   To restart: python run_analysis.py\n")


if __name__ == "__main__":
    main()
