from shapely.geometry import LineString
from shapely.strtree import STRtree
import numpy as np

def intersCPU_shapely(A, B):
    A = np.asarray(A)
    B = np.asarray(B)

    # transpose if MATLAB style (2,N)
    if A.shape[0] == 2 and A.shape[1] != 2:
        A = A.T
    if B.shape[0] == 2 and B.shape[1] != 2:
        B = B.T

    # segments
    segsA = [LineString([tuple(A[i]), tuple(A[i+1])]) for i in range(A.shape[0]-1)]
    segsB = [LineString([tuple(B[i]), tuple(B[i+1])]) for i in range(B.shape[0]-1)]

    # build spatial index for B
    treeB = STRtree(segsB)

    out = []
    id1 = []
    id2 = []

    for i, segA in enumerate(segsA):
        # query returns indices in Shapely 1.x
        cand_idx = treeB.query(segA)
        for j in np.atleast_1d(cand_idx):
            segB = segsB[int(j)]
            inter = segA.intersection(segB)
            if inter.is_empty:
                continue
            if inter.geom_type == "Point":
                out.append((inter.x, inter.y))
                id1.append(i)
                id2.append(int(j))
            elif inter.geom_type == "MultiPoint":
                for p in inter.geoms:
                    out.append((p.x, p.y))
                    id1.append(i)
                    id2.append(int(j))
            elif inter.geom_type == "LineString":
                for x,y in inter.coords:
                    out.append((x,y))
                    id1.append(i)
                    id2.append(int(j))

    return np.array(out), np.array(id1), np.array(id2)




def intersCPU_numpy(A, B):
    """
    Compute intersections between two polylines using NumPy only.
    A, B: arrays of shape (N,2)
    Returns:
        out  : Nx2 array of intersection points
        id1  : indices of segments in A
        id2  : indices of segments in B
        t1   : parametric position along segment in A
        t2   : parametric position along segment in B
    """
    A = np.asarray(A)
    B = np.asarray(B)

    L1 = A.shape[0]
    L2 = B.shape[0]

    if L1 < 2 or L2 < 2:
        return np.empty((0,2)), np.array([]), np.array([]), np.array([]), np.array([])

    # precompute segment bounding boxes
    xmin1 = np.minimum(A[:-1,0], A[1:,0]); xmax1 = np.maximum(A[:-1,0], A[1:,0])
    ymin1 = np.minimum(A[:-1,1], A[1:,1]); ymax1 = np.maximum(A[:-1,1], A[1:,1])

    xmin2 = np.minimum(B[:-1,0], B[1:,0]); xmax2 = np.maximum(B[:-1,0], B[1:,0])
    ymin2 = np.minimum(B[:-1,1], B[1:,1]); ymax2 = np.maximum(B[:-1,1], B[1:,1])

    out = []
    id1 = []
    id2 = []
    t1 = []
    t2 = []

    for i in range(L1-1):
        # candidate B segments whose bounding boxes overlap
        mask = (xmax1[i] >= xmin2) & (xmin1[i] <= xmax2) & (ymax1[i] >= ymin2) & (ymin1[i] <= ymax2)
        if not np.any(mask):
            continue

        dx1 = A[i+1,0] - A[i,0]
        dy1 = A[i+1,1] - A[i,1]

        for j in np.where(mask)[0]:
            dx2 = B[j+1,0] - B[j,0]
            dy2 = B[j+1,1] - B[j,1]

            denom = dx1*dy2 - dy1*dx2
            if np.abs(denom) < 1e-12:
                continue  # parallel segments

            dx = B[j,0] - A[i,0]
            dy = B[j,1] - A[i,1]

            s = (dx*dy2 - dy*dx2)/denom
            t = (dx*dy1 - dy*dx1)/denom

            if 0 < s < 1 and 0 < t < 1:
                px = A[i,0] + s*dx1
                py = A[i,1] + s*dy1
                out.append([px, py])
                id1.append(i)
                id2.append(j)
                t1.append(s)
                t2.append(t)

    return (np.array(out), np.array(id1), np.array(id2), np.array(t1), np.array(t2))
