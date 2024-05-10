import os
import sys

current_dir = os.path.dirname(__file__)
project_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_dir)
