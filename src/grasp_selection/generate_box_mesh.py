import IPython
import mesh
import obj_file
import os
import sys

import mayavi.mlab as mv

if __name__ == '__main__':
    w = float(sys.argv[1])
    h = float(sys.argv[2])
    d = float(sys.argv[3])
    out_filename = sys.argv[4]

    x_off = 0
    y_off = 0
    z_off = 0
    if len(sys.argv) > 5:
        x_off = float(sys.argv[5])
        y_off = float(sys.argv[6])
        z_off = float(sys.argv[7])

    x_coords = [x_off-w/2, x_off+w/2]
    y_coords = [y_off-h/2, y_off+h/2]
    z_coords = [z_off-d/2, z_off+d/2]
    points = []

    for x in x_coords:
        for y in y_coords:
            for z in z_coords:
                points.append([x, y, z])

    tris = [
        [0, 1, 2],
        [1, 3, 2],
        [6, 5, 4],
        [6, 7, 5],
        [2, 3, 6],
        [3, 7, 6],
        [4, 1, 0],
        [4, 5, 1],
        [0, 2, 4],
        [2, 6, 4],
        [5, 3, 1],
        [5, 7, 3]
        ]

    m = mesh.Mesh3D(points, tris)
    m.visualize(style='wireframe')
    mv.axes()
    mv.show()

    objf = obj_file.ObjFile(out_filename)
    objf.write(m)
