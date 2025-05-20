import numpy as np


def check_collision(p0, p1, blocks):
  """
  Check if a line segment from p0 to p1 intersects with the given AABB block.
  block: [xmin, ymin, zmin, xmax, ymax, zmax]
  """
  assert len(p0) == 3 and len(p1) == 3, "p0 and p1 must be 3D points"
  assert len(blocks) == 6, "block must be a 6D array"
  
  p0 = np.array(p0, dtype=np.float32)
  p1 = np.array(p1, dtype=np.float32)
  d = p1 - p0
  tmin, tmax = 0.0, 1.0

  # check for x, y, z axis
  for i in range(3): 
      if abs(d[i]) < 1e-8:
          if p0[i] < blocks[i] or p0[i] > blocks[i + 3]:
              return False
      else:
          t1 = (blocks[i]     - p0[i]) / d[i]
          t2 = (blocks[i + 3] - p0[i]) / d[i]
          t_enter, t_exit = min(t1, t2), max(t1, t2)
          tmin = max(tmin, t_enter)
          tmax = min(tmax, t_exit)
          # if tmin > tmax, no intersection
          if tmin > tmax:
              return False
  return True

