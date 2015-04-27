import openravepy as orv
import numpy as np
import IPython

import mesh
import obj_file as of

if __name__ == '__main__':
    env = orv.Environment()
    env.SetViewer('qtcoin')
    env.Load('robots/barretthand.robot.xml')
    robot = env.GetRobots()[0]

    i = 0
    for link in robot.GetLinks():
        link_geom = link.GetGeometries()[0]
        link_tris = link_geom.GetCollisionMesh()

        verts = link_tris.vertices.tolist()
        tris = link_tris.indices.tolist()
        m = mesh.Mesh(verts, tris)

        filename = 'link_%d.obj' %(i)
        obj_f = of.ObjFile(filename)
        obj_f.write(m)

        print 'Saved link', i
        i = i+1


