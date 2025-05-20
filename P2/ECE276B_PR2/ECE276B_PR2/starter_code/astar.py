
# priority queue for OPEN list
from collections import defaultdict
import numpy as np
from pqdict import pqdict
import math
from utils import check_all_blocks, is_in_boundary

class AStarNode(object):
  def __init__(self, key):
    self.key = key
    self.g = math.inf
    self.f = math.inf
    self.h = math.inf
    self.parent_node = None
    self.is_open = False
    self.closed = False

  def __lt__(self, other):
    return self.g < other.g 

  def setParent(self,parent):
    self.parent_node = parent


  def setG(self, g):
    self.g = g
    self.f = self.g + self.h

  def setHeuristic(self, h, epsilon):
    self.h = h * epsilon
    self.f = self.g + self.h

  def getHeuristic(self):
    return self.h
  
  def isOpen(self):
    return self.is_open
  
  def isClosed(self):
    return self.closed
  
  def getG(self):
    return self.g
  
  def getF(self):
    self.f = self.g + self.h
    return self.f
  
  def getParent(self):
    return self.parent_node
  
  

class AStar(object):
  def __init__(self, start, goal, blocks, boundary, epsilon = 1, map_resolution=0.5, edge_cost=1.0):
    self.map_resolution = map_resolution
    #self.start = self.snap_point(start, map_resolution)
    #self.goal = self.snap_point(goal, map_resolution)
    self.start = tuple(round(coord, 1) for coord in start)
    self.goal = tuple(round(coord, 1) for coord in goal)
    self.blocks = blocks
    self.boundary = boundary
    self.epsilon = epsilon
    self.edge_cost = edge_cost

  @staticmethod
  def snap_point(p, map_resolution):
    return tuple(round(round(coord / map_resolution) * map_resolution, 1) for coord in p)


  def initialize(self):
    open_heap = pqdict().minpq()
    nodes = {}
    # set up start node
    start_node = AStarNode(self.start)
    start_node.setHeuristic(self.compute_heuristic(self.start, self.goal), self.epsilon)
    start_node.setG(0)
    start_node.is_open = True
    f = start_node.getF()
    open_heap[self.start] = (f, self.start)
    nodes[self.start] = start_node
    return open_heap, nodes
  

  def generate_neighbors(self, node_key):
    # generate neighbors for a node
    offsets = []
    for dx in [-self.map_resolution, 0, self.map_resolution]:
        for dy in [-self.map_resolution, 0, self.map_resolution]:
            for dz in [-self.map_resolution, 0, self.map_resolution]:
                if dx == dy == dz == 0:
                    continue
                offsets.append((dx, dy, dz))
    # generate neighbors
    for dx, dy, dz in offsets:
        neighbor = (round(node_key[0] + dx, 1), round(node_key[1] + dy, 1), round(node_key[2] + dz, 1))
        # check boundary
        if not is_in_boundary(neighbor, self.boundary):
            continue
        # check collision
        if check_all_blocks(node_key, neighbor, self.blocks):
            continue
        yield neighbor, self.edge_cost
    
    # check if can reach goal
    if np.linalg.norm(np.array(node_key) - np.array(self.goal)) <= self.map_resolution:
      if not check_all_blocks(node_key, self.goal, self.blocks):
          yield self.goal, self.edge_cost


  def find_node(self, nodes, key):
    return nodes.get(key)
    
  @staticmethod
  def compute_heuristic(p1,p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    # Euclidean distance
    return np.linalg.norm(p1 - p2)
  
  def get_path(self, nodes, node_key):
    path = []
    current = nodes[node_key]
    while current:
      path.append(np.array(current.key))
      current = current.getParent()
    return np.array(list(reversed(path)))
  

  def is_goal(self, node_key):
    #return np.linalg.norm(np.array(node_key) - np.array(self.goal)) <= np.sqrt(0.1)
    return node_key == self.goal


  def plan(self):
    self.open_heap, self.nodes = self.initialize()
    self.closed_list = []
    
    while self.open_heap:
      item = self.open_heap.popitem()
      current_key = item[0]
      current_node = self.find_node(self.nodes, current_key)
      current_node.is_open = False
      current_node.closed = True
      self.closed_list.append(current_key)
      
      # check if reached goal
      if self.is_goal(current_key):
        return self.get_path(self.nodes, current_key)
      
      # find neighbors
      for child_key, edge_cost in self.generate_neighbors(current_key):
        # check if child node is in nodes
        child_node = None
        if child_key not in self.nodes:
          child_node = AStarNode(child_key)
          child_node.setHeuristic(self.compute_heuristic(child_key, self.goal), self.epsilon)
          self.nodes[child_key] = child_node
        else:
          child_node = self.find_node(self.nodes, child_key)
        # check if child node is in closed list
        if child_node.isClosed():
          continue
        tentative_g = current_node.getG() + edge_cost
        if tentative_g < child_node.getG():
          child_node.setG(tentative_g)
          child_node.setParent(current_node)
          child_node.is_open = True
          f = child_node.getF()
          self.open_heap[child_key] = (f, child_key)

    return None
          



