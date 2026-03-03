"""
Pytest configuration 
src.* and configs.* imports resolve correctly from any working directory.
"""
import sys
import os

# Insert federated_learning/ directory (one level above tests/) into path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))