from collections import defaultdict
import numpy as np
from utils import is_in_collision


class Graph:
    """
    Build a graph from the boundary and blocks.
    """
    # NOTE: Not used anymore
    
    def __init__(self, boundary, blocks):
        self.boundary = boundary
        self.blocks = blocks


    def snap_point_to_nearest_valid(self, p):
        """Find the closest vertex in self.vertices_set to point p."""
        p = np.array(p)
        min_dist = float('inf')
        nearest = None
        for v in self.vertices:
            dist = np.linalg.norm(p - np.array(v))
            if dist < min_dist:
                min_dist = dist
                nearest = v
        return nearest


    def generate_vertices(self, resolution=0.5):
        """Generate grid vertices within boundary, optionally avoiding blocks."""
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary[0][:6]
        x_vals = np.arange(xmin, xmax + 1e-4, resolution)
        y_vals = np.arange(ymin, ymax + 1e-4, resolution)
        z_vals = np.arange(zmin, zmax + 1e-4, resolution)

        vertices_set = set()

        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    p = (round(x, 1), round(y, 1), round(z, 1))
                    if not is_in_collision(p, self.blocks):
                        vertices_set.add(p)
        self.vertices = list(vertices_set)
        return self.vertices
    
    def build_edges(self, resolution=0.5, connection_type='26'):
        """Create lazy-check adjacency graph with 6 or 26 connectivity."""
        assert connection_type in ['6', '26'], "Only '6' or '26' allowed"
        offsets = []
        for dx in [-resolution, 0, resolution]:
            for dy in [-resolution, 0, resolution]:
                for dz in [-resolution, 0, resolution]:
                    if dx == dy == dz == 0:
                        continue
                    if connection_type == '6' and sum(map(abs, [dx, dy, dz])) != resolution:
                        continue
                    offsets.append((dx, dy, dz))

        self.edges = defaultdict(dict)
        for v in self.vertices:
            for dx, dy, dz in offsets:
                neighbor = (round(v[0] + dx, 1), round(v[1] + dy, 1), round(v[2] + dz, 1))
                if neighbor in self.vertices:
                    self.edges[v][neighbor] = 1.0  # lazy check later

        return self.edges

    def connect_point_to_graph(self, point, resolution=0.5, connection_type='26'):
        """Explicitly add a point and connect it to valid neighbors."""
        if is_in_collision(point, self.blocks):
            raise ValueError(f"Point {point} is inside an obstacle!")
        
        point = tuple(round(coord, 1) for coord in point)
        # check if point is already in graph
        if point in self.vertices:
            point_incoming = any(point in nbrs for nbrs in self.edges.values())
            point_outgoing = point in self.edges
            if point_incoming or point_outgoing:
                return
                    
        # find the nearest valid vertex
        nearest_vertex = self.snap_point_to_nearest_valid(point)
        has_incoming = any(nearest_vertex in nbrs for nbrs in self.edges.values())
        has_outgoing = nearest_vertex in self.edges
        # connect nearest vertex to graph if it is not already connected
        if not has_incoming and not has_outgoing:
            offsets = []
            for dx in [-resolution, 0, resolution]:
                for dy in [-resolution, 0, resolution]:
                    for dz in [-resolution, 0, resolution]:
                        if dx == dy == dz == 0:
                            continue
                        if connection_type == '6' and sum(map(abs, [dx, dy, dz])) != resolution:
                            continue
                        offsets.append((dx, dy, dz))

            for dx, dy, dz in offsets:
                neighbor = (round(nearest_vertex[0] + dx, 1), round(nearest_vertex[1] + dy, 1), round(nearest_vertex[2] + dz, 1))
                if neighbor in self.vertices:
                    self.edges[nearest_vertex][neighbor] = 1.0
                    self.edges[neighbor][nearest_vertex] = 1.0

        # connect point to graph
        if point not in self.vertices:
            self.vertices.append(point)

        self.edges[nearest_vertex][point] = 1.0
        self.edges[point][nearest_vertex] = 1.0

    def build_graph(self, start, goal, resolution=0.5, connection_type='26'):
        """Build a graph from start to goal."""
        self.generate_vertices(resolution)
        self.build_edges(resolution, connection_type)
        self.connect_point_to_graph(start, resolution, connection_type)
        self.connect_point_to_graph(goal, resolution, connection_type)
        return self.vertices, self.edges
        
