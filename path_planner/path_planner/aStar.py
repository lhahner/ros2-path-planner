"""
"""
class AStar(Planner):
    def __init__(self, map, threshold):
        """
        """
        self.path = None
        self.map = map
        self.threshold = threshold
        self.cost_map = 255.0 / np.maximum(self.map, 1)
        self.height, self.width = map.shape

    def heurstic(self, a, b):
        """
        """
        return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    def get_neighbors(self, node, treshold):
        r, c = node
        
       directions = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)] # move sidewards or diagonal

        neighbors = []
        for r, c in directions:
            if 0 <= r < self.height and 
               0 <= c < self.width and self.map[r,c] > treshold: 
                neighbors.append((r, c))

        return neighbors

    def plan(self, start, goal):
        """
        """
        visited_nodes = set()
        adjacent_visited_nodes = set()
        visited_nodes.add(start)

        came_from_nodes = {}
        g_score = {start: 0}
        f_score = {start: self.heurstic(start, goal)}

        while visited_nodes:
            current = min(visited_node, key=lambda x: f_score[x])
            if current == goal:
                while current in came_from_nodes:
                    self.path.append(current)
                    current = came_from_nodes[current]
                return path[::-1]
        visited_node.remove(current)
        adjacent_visited_nodes.add(current)

        for neighbor in self.get_neighbors(current, threshold):
            tentative_g_score = g_score[current] + self.cost_map[neighbor[0],neighbor[1]]
            if neighbor in closedset and tentative_g_score >= g_score[neighbor]:
                continue

            if neighbor not in visited_nodes or tentative_g_score < g_score[neighbor]:
                came_from_nodes[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)

                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
        return path
