import math

class Node:
    def __init__(self, x, y, g=float('inf'), h=0.0, parent=None):  # default g = inf
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.parent = parent

    @property
    def f(self):
        return self.g + self.h

    def __eq__(self, other):
        return isinstance(other, Node) and self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash((self.x, self.y))

    def distance_to(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def __repr__(self):
        return f"Node(x={self.x}, y={self.y}, g={self.g:.2f}, h={self.h:.2f}, f={self.f:.2f})"
