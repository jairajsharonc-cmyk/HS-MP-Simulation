"""HS-MP: Homogeneous Structured Multi-Path P2P video-streaming testbed.

Clean-room implementation of the testbed described in:
  "Machine Learning versus Heuristics in Peer-to-Peer Video Streaming:
   A Statistical Evaluation of Predictive Caching and Adaptive Bitrate Control"
"""
from .config import Config, CONFIGS
from .simulator import run_simulation

__all__ = ["Config", "CONFIGS", "run_simulation"]
