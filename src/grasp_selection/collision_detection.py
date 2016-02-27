from numpy import cross, dot, array, pi, ndarray
from mesh import Mesh3D
from obj_file import ObjFile
from similarity_tf import SimilarityTransform3D as tf
from tfx import pose, rotation_tb

class CollisionDetection:    
    
    _EPSILON = 1e-6
    
    @staticmethod
    def _triangle_to_edges(t):
        vs = [v for v in t]
        return ((vs[0], vs[1]), (vs[0], vs[2]), (vs[1], vs[2]))
    
    @staticmethod
    def collide_triangle_triangle(t1, t2):
        edges = CollisionDetection._triangle_to_edges(t2)
        
        for edge in edges:
            if CollisionDetection.collide_trianlge_seg(t1, edge):
                return True
                
        return False
        
    @staticmethod
    def _cast_to_np_array(v):
        if type(v) is not ndarray:
            v = array(v)
        return v
        
    @staticmethod
    def collide_trianlge_seg(triangle, seg):
        vs = [CollisionDetection._cast_to_np_array(v) for v in triangle]
        
        v1, v2, v3 = vs[0], vs[1], vs[2]
        e1 = v2 - v1
        e2 = v3 - v1        
        
        O = seg[0]
        D = seg[1]
        
        P = cross(D, e2)
        det = dot(e1, P)
        if det > -CollisionDetection._EPSILON and det < CollisionDetection._EPSILON:
            return False
        
        inv_det = 1./det
        T = O - v1
        u = dot(T, P) * inv_det
        if u < 0 or u > 1:
            return False
            
        Q = cross(T, e1)
        v = dot(D, Q) * inv_det
        if v < 0 or v > 1:
            return False
            
        t = dot(e2, Q) * inv_det
        if t > CollisionDetection._EPSILON:
            return True
            
        return False
        
    @staticmethod
    def collide_mesh_mesh(m1, m2):
        m1_triangles = m1.triangles()
        m2_triangles = m2.triangles()
        print "Triangles count: m1:{0}, m2:{1}".format(len(m1_triangles), len(m2_triangles))
        for t1 in m1_triangles:
            for t2 in m2_triangles:
                _t1 = [m1.vertices()[t1[i]] for i in range(3)]
                _t2 = [m2.vertices()[t2[i]] for i in range(3)]
                if CollisionDetection.collide_triangle_triangle(_t1, _t2):
                    return True
        return False
        
def test_collision_detection():
    '''
    print "Performing CollisionDetection Unit Tests..."
    v1 = array([0, 0, 0])
    v2 = array([1, 0, 0])
    v3 = array([0, 1, 0])
    t1 = (v1, v2, v3)
    
    print "Testing segment collides with triangle..."
    p1 = array([0.5, 0.5, 0.5])
    p2 = array([0.5, 0.5, -0.5])
    s1 = (p1, p2)
    assert CollisionDetection.collide_trianlge_seg(t1, s1), "Error! Triangle {0} is supposed to collide with line {1}".format(t1, s1)
    print "Passed!"
    
    print "Testing segment does not collide with triangle..."
    p3 = array([1, 1, 1])
    p4 = array([1, 1, -1])
    s2 = (p3, p4)
    assert not CollisionDetection.collide_trianlge_seg(t1, s2), "Error! Triangle {0} is not supposed to collide with line {1}".format(t1, s2)
    print "Passed!"
    
    print "Testing triangle collides with triangle..."
    v4 = array([0.5, 0.5, 1])
    v5 = array([0,0,-1])
    v6 = array([1,0,-1])
    t2 = (v4, v5, v6)
    assert CollisionDetection.collide_triangle_triangle(t1, t2), "Error! Triangle {0} is supposed to collide with triangle {1}".format(t1, t2)
    print "Passed!"
    
    print "Testing triangle does not collide with triangle..."
    v7 = array([0, -0.5, 0])
    v8 = array([1, -0.5, 0])
    v9 = array([0.5, -0.5, 1])
    t3 = (v7, v8, v9)
    assert not CollisionDetection.collide_triangle_triangle(t1, t3), "Error! Triangle {0} is not supposed to collide with triangle {1}".format(t1, t3)
    print "Passed!"
    
    print "Testing mesh collides with mesh..."
    z_offset = array([0,0,1])
    w1 = v1 + z_offset
    m1 = Mesh3D([v1, v2 ,v3, w1], [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    w2 = array([0,0,1])
    m2 = Mesh3D([v4, v5, v6, w2], [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2 ,3]])
    assert CollisionDetection.collide_mesh_mesh(m1, m2), "Error! Mesh {0} is supposed to collide with mesh {1}".format(m1, m2)
    print "Passed!"
    
    print "Testing mesh does not collide with mesh..."
    y_offset = array([0, -1, 0])
    w3 = v9 + y_offset
    m3 = Mesh3D([v7, v8, v9, w3], [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    assert not CollisionDetection.collide_mesh_mesh(m1, m3), "Error! Mesh {0} is not supposed to collide with mesh {1}".format(m1, m3)    
    print "Passed!"
    '''
    
    
    obj_path = "/mnt/terastation/shape_data/SHREC14LSGTB/M005130.obj"
    of = ObjFile(obj_path)
    m4 = of.read()
    m5 = of.read()
    
    print "Testing real mesh collide with mesh..."
    rot_tf = tf(pose([0,0,0], rotation_tb(pi/4,0,0)))
    m5_rot = m5.transform(rot_tf)
    assert CollisionDetection.collide_mesh_mesh(m4, m5_rot), "Error! Mesh {0} is supposed to collide with mesh {1}".format(m4, m5_rot)
    print "Passed!"
    
    print "Testing real mesh does not collide with mesh..."
    tra_tf = tf(pose([10,10,10]))
    m5_tra = m5.transform(tra_tf)
    assert CollisionDetection.collide_mesh_mesh(m4, m5_tra), "Error! Mesh {0} is not supposed to collide with mesh {1}".format(m4, m5_tra)
    
    print "All tests passed!"
        
if __name__ == "__main__":
    test_collision_detection()
        
        