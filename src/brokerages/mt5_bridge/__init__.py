"""
MT5 Bridge - Cross-Platform MetaTrader 5 Access
================================================

Allows running the AI Trade Bot on Linux while the MT5 terminal runs on Windows.

Architecture:
    [Linux: AI Trade Bot]  <--TCP-->  [Windows: MT5 Bridge Server]
         MT5BridgeClient                    MT5BridgeServer

Components:
- protocol.py: JSON message format shared between client and server
- server.py: Windows-side TCP server wrapping MetaTrader5 Python package
- client.py: Linux-side TCP client proxying MT5 calls

Usage (Windows - run the server):
    python -m src.brokerages.mt5_bridge.server --port 5555

Usage (Linux - the client is used automatically):
    # MT5DataProvider and MT5Brokerage auto-detect the platform
    # and use MT5BridgeClient when not on Windows
"""
