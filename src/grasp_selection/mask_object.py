import argparse
import os
import numpy as np

import obj_file as of
import mesh as m

def load_mask_indices(mask_fname):
    return np.load(mask_fname)

def compute_unmasked_region(mesh, bitmask):
    """Return a mesh without the masked faces."""
    unmasked = m.Mesh3D(mesh.vertices(), mesh.triangles())
    verts = unmasked.vertices()
    tris = np.array(unmasked.triangles())
    new_tris = tris[~bitmask].tolist()

    # Remove masked tris and resulting unreferenced vertices
    unmasked.set_triangles(new_tris)
    unmasked.remove_unreferenced_vertices()
    return unmasked

def compute_masked_region(mesh, bitmask):
    """Return a mesh with the masked faces as a convex hull."""
    masked = m.Mesh3D(mesh.vertices(), mesh.triangles())
    verts = masked.vertices()
    tris = np.array(masked.triangles())
    new_tris = tris[bitmask].tolist()

    # Remove unmasked tris and resulting unreferenced vertices
    masked.set_triangles(new_tris)
    masked.remove_unreferenced_vertices()
    return masked.convex_hull()

def zipper(unmasked, masked):
    """Return a mesh that combines the masked and unmasked meshes."""
    verts_unmasked, tris_unmasked = unmasked.vertices(), unmasked.triangles()
    verts_masked, tris_masked = masked.vertices(), masked.triangles()

    # TODO: check for duplicate vertices
    offset = len(verts_unmasked)
    new_verts = verts_unmasked + verts_masked
    new_tris = tris_unmasked
    for tri in tris_masked:
        new_tri = [v + offset for v in tri]
        new_tris.append(new_tri)
    return m.Mesh3D(new_verts, new_tris)

def mask(mesh, mask_indices):
    """Return a mesh where the masked faces are replaced with their convex hull.

    mesh -- mesh.Mesh3D
    mask_indices -- np.array of face indices (0-indexed)
    """
    bitmask = np.zeros(len(mesh.triangles()), np.bool)
    bitmask[mask_indices] = 1

    unmasked = compute_unmasked_region(mesh, bitmask)
    masked = compute_masked_region(mesh, bitmask)
    merged = zipper(unmasked, masked)
    return merged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('obj_file')
    args = parser.parse_args()

    # Read mesh to be masked
    mesh = of.ObjFile(args.obj_file).read()

    obj_dir, obj_fname = os.path.split(args.obj_file)
    obj_name = obj_fname.split('.')[0]

    # Load mask indices
    mask_fname = '{}_mask.npy'.format(obj_name)
    mask_indices = load_mask_indices(os.path.join(obj_dir, mask_fname))

    # Apply mask to create new object, write masked object to *_masked.obj
    masked = mask(mesh, mask_indices)
    masked_obj_fname = '{}_masked.obj'.format(obj_name)
    of.ObjFile(os.path.join(obj_dir, masked_obj_fname)).write(masked)

if __name__ == '__main__':
    main()
