"""
Script to rescale meshes for easy imports
Author: Jeff Mahler
"""
import logging
import os
import sys

import mesh
import obj_file as objf

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    in_mesh_filename = sys.argv[1]
    out_mesh_filename = sys.argv[2]    
    scale_factor = float(sys.argv[3])

    in_of = objf.ObjFile(in_mesh_filename)
    m = in_of.read()
    m.rescale(scale_factor)

    out_of = objf.ObjFile(out_mesh_filename)
    out_of.write(m)
