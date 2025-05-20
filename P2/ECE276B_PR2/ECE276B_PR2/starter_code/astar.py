
# priority queue for OPEN list
import numpy as np
from pqdict import pqdict
import math
from utils import check_all_blocks

class AStarNode(object):
  def __init__(self, key):
    self.key = key
    self.g = math.inf
    self.f = math.inf
    self.h = math.inf
    self.parent_node = None
    self.children = []
    self.is_open = False
    self.closed = False

  def __lt__(self, other):
    return self.g < other.g 

  def setParent(self,parent):
    self.parent_node = parent

  def setChildren(self,children):
    self.children = children

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
  
  def getChildren(self):
    return self.children
  

class AStar(object):
  def __init__(self, start, goal, blocks, edges, vertices, epsilon = 1):
    self.start = tuple(round(coord, 1) for coord in start)
    self.goal = tuple(round(coord, 1) for coord in goal)
    self.blocks = blocks
    self.edges = edges
    self.vertices = vertices
    self.epsilon = epsilon

  def initialize(self):
    open_heap = pqdict().minpq()
    nodes = {}
    for v in self.vertices:
      node = AStarNode(v)
      node.setHeuristic(self.compute_heuristic(v, self.goal), self.epsilon)
      node.setChildren(list(self.edges[v].keys()))
      if v == self.start:
        node.setG(0)
        node.is_open = True
        f = node.getF()
        open_heap[v] = (f, v)
      nodes[v] = node
    return open_heap, nodes
  
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
      
      if current_key == self.goal:
        return self.get_path(self.nodes, current_key)
      
      for child_key, edge_cost in self.edges[current_key].items():
        child_node = self.find_node(self.nodes, child_key)
        # check if child node is in closed list
        if child_node.isClosed():
          continue
        
        # check collision
        if check_all_blocks(current_node.key, child_node.key, self.blocks):
          continue
        
        tentative_g = current_node.getG() + edge_cost
        if tentative_g < child_node.getG():
          child_node.setG(tentative_g)
          child_node.setParent(current_node)
          child_node.is_open = True
          f = child_node.getF()
          self.open_heap[child_key] = (f, child_key)

    return None
          



