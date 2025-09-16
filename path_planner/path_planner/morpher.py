from .imports import *

class Morpher:
    def __init__(self, occupancy_grid, robot_width):
        self.occupancy_grid = occupancy_grid
        self.robot_width = robot_width

    def _grid_to_np(self, msg):
        h, w = msg.info.height, msg.info.width
        a = np.asarray(msg.data, dtype=np.int8).reshape((h, w))
        return a

    def _np_to_grid(self, arr, tmpl):
        out = OccupancyGrid()
        out.header = tmpl.header
        out.info = tmpl.info
        out.data = arr.astype(np.int8).flatten().tolist()
        return out

    def _to_cv_mask(self, a, occ_thresh=75, unknown_as_occ=True):
        if unknown_as_occ:
            a = np.where(a < 0, 100, a)
        mask = (a >= occ_thresh).astype(np.uint8) * 255
        return mask

    def _mask_to_occ(self, mask, unknown_fill=-1):
        # map back to 0/100 and keep unknown as unknown_fill
        occ = (mask > 0).astype(np.int8) * 100
        if unknown_fill != -1:
            return occ
        # restore unknown where original was unknown
        a_orig = self._grid_to_np(self.occupancy_grid)
        unknown = (a_orig < 0)
        occ[unknown] = -1
        return occ

    def dilate(self):
        assert self.occupancy_grid is not None, "grid is empty"
        a = self._grid_to_np(self.occupancy_grid)
        print(a.shape)
        mask = self._to_cv_mask(a, occ_thresh=50, unknown_as_occ=False)
        kernel = np.ones((self.robot_width, self.robot_width), np.uint8)
        dil = cv.dilate(mask, kernel, iterations=1)
        occ_arr = self._mask_to_occ(dil, unknown_fill=-1)
             
        #plt.imshow(occ_arr)
        #plt.savefig("test.png")
       # plt.close()
        return self._np_to_grid(occ_arr, self.occupancy_grid)
