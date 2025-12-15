#!/usr/bin/env python3
"""
Test Notification System
========================

Test all notification channels work properly.

Usage:
    python scripts/test_notifications.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

from src.notifier import Notifier


def main():
    print("\n" + "="*60)
    print("TESTING NOTIFICATION SYSTEM")
    print("="*60 + "\n")

    notifier = Notifier()

    # Show status
    status = notifier.get_status()
    print("Current Configuration:")
    print(f"  Desktop notifications: {'✅ Enabled' if status['desktop_enabled'] else '❌ Disabled'}")
    print(f"  Sound alerts: {'✅ Enabled' if status['sound_enabled'] else '❌ Disabled'}")
    print(f"  Telegram: {'✅ Enabled' if status['telegram_enabled'] else '❌ Disabled'}")

    if status['telegram_enabled'] and not status['telegram_configured']:
        print("    ⚠️  Telegram enabled but not configured (missing token/chat_id)")

    print("\n" + "-"*60)
    print("Sending test notification...")
    print("-"*60 + "\n")

    # Send test
    notifier.test_notifications()

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("\nIf you saw:")
    print("  ✅ Desktop notification popup -> Desktop alerts work")
    print("  ✅ Sound alert -> Sound notifications work")
    print("  ✅ Console output above -> Console logging works")
    print("\nIf something didn't work, check:")
    print("  - Desktop: Install 'plyer' or enable system notifications")
    print("  - Sound: Check system volume and audio drivers")
    print("  - Telegram: Add bot_token and chat_id in config.yaml")


if __name__ == "__main__":
    main()
