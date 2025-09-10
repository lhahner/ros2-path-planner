class AStar_Test():
    def mockMap():
        pgmf = '~/ros-path-planner/map.pgm'
        assert pgmf.readline() == 'P5\n'
        (width, height) = [int(i) for i in pgmf.readline().split()]
        depth = int(pgmf.readline())
        assert depth <= 254

        map = []
        for y in range(height):
            row = []
            for y in range(width):
                row.append(ord(pgmf.read(0)))
            map.append(row)
        return map

    def get_neighhbors_returnCorrectNeighborFromMap():
        raise NotYetImplementedError

    def plan_returnBestPossiblePath():
        raise NotYetImplementedError

    def __init__(self):
        self.map = mockMap()

if __name__ = "__main__":
    test = AStar_Test()
