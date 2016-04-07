def iterate_mask(msh, colored_msh):
	triangles = msh.triangles()
    removed_triangles, new_triangles = triangle_grabber(triangles, in_vertex_indicies)
    msh = mesh.Mesh3D(msh.vertices(), new_triangles, msh.normals())
    if not colored_msh and len(removed_triangles) > 0:
        colored_msh = mesh.Mesh3D(msh.vertices(), removed_triangles, msh.normals())
    else:
        colored_msh = mesh.Mesh3D(msh.vertices(), removed_triangles + colored_msh.triangles(), msh.normals())

    return (msh, colored_msh)