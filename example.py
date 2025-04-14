import numpy as np

forward = np.array([-10, 0, -10])
forward = forward / np.linalg.norm(forward)
print(forward)

up = np.array([0, 1, 0])
right = np.cross(up, forward)
print(right)
up = np.cross(forward, right)