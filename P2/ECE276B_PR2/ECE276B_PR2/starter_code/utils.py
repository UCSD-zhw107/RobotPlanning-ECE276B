import numpy as np


def check_all_blocks(p0, p1, blocks):
    for block in blocks:
        if check_collision(p0, p1, block[:6]):
            return True
    return False

def check_collision(p0, p1, block):
  """
  Check if a line segment from p0 to p1 intersects with the given AABB block.
  block: [xmin, ymin, zmin, xmax, ymax, zmax]
  """
  assert len(p0) == 3 and len(p1) == 3, "p0 and p1 must be 3D points"
  assert len(block) == 6, "block must be a 6D array"
  
  p0 = np.array(p0, dtype=np.float32)
  p1 = np.array(p1, dtype=np.float32)
  d = p1 - p0
  tmin, tmax = 0.0, 1.0

  # check for x, y, z axis
  for i in range(3): 
      if abs(d[i]) < 1e-8:
          if p0[i] < block[i] or p0[i] > block[i + 3]:
              return False
      else:
          t1 = (block[i]     - p0[i]) / d[i]
          t2 = (block[i + 3] - p0[i]) / d[i]
          t_enter, t_exit = min(t1, t2), max(t1, t2)
          tmin = max(tmin, t_enter)
          tmax = min(tmax, t_exit)
          # if tmin > tmax, no intersection
          if tmin > tmax:
              return False
  return True

def is_in_collision(p, blocks):
    """Check if point p is inside any AABB block."""
    for block in blocks:
        if (block[0] <= p[0] <= block[3] and
            block[1] <= p[1] <= block[4] and
            block[2] <= p[2] <= block[5]):
            return True
    return False


def is_in_boundary(p, boundary):
    """Check if point p is inside the boundary."""
    xmin, ymin, zmin, xmax, ymax, zmax = boundary[0][:6]
    # outside boundary
    if not (xmin <= p[0] <= xmax and ymin <= p[1] <= ymax and zmin <= p[2] <= zmax):
        return False
    return True
