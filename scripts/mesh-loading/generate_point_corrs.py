import os
import numpy as np
import sys

import obj_file
import mesh
import tfx

import IPython

argc = len(sys.argv)
filename = sys.argv[1]

file_root, file_ext = os.path.splitext(filename)
of = obj_file.ObjFile(filename)
m = of.read()

tf = tfx.random_tf()

v_array = np.array(m.get_vertices())
w_array = np.array(m.get_vertices())
w_list = []
for i in range(v_array.shape[0]):
    w = np.array(tf.apply(v_array[i,:]))
    w = w.T[0]

    w_array[i,:] = w
    w_list.append([w[0], w[1], w[2]])

m_t = mesh.Mesh(w_list, m.get_triangles())
new_filename = file_root + '_tf' + file_ext

of_out = obj_file.ObjFile(new_filename)
of_out.write(m_t)

np.savetxt('tf.txt', tf.matrix)
np.savetxt(file_root + '_pts.txt', v_array)
np.savetxt(file_root + '_pts_tf.txt', w_array)
