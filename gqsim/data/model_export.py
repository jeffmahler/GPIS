__author__ = 'Junggon Kim'
import os
import time
import sys
from numpy import *
from optparse import OptionParser
from openravepy import *

def run():
    parser = OptionParser(description='export OpenRAVE model information.')
    parser.add_option('--fi',
                      action="store",type='string',dest='openravemodelfilename',default=None,
                      help='')
    parser.add_option('--fo',
                      action="store",type='string',dest='outputfilename',default=None,
                      help='')
    (options, args) = parser.parse_args()

    # create openrave environment
    env = Environment()
    env.SetViewer('qtcoin')
    env.Load(os.path.abspath(options.openravemodelfilename))
    
    # ground geometries
    ground_link = []
    ground_link_T = []
    
    # export
    f = open(options.outputfilename, 'w')
      
    f.write('<world name = "%s">\n\n'%options.outputfilename)
    for kinbody in env.GetBodies():
        
        T = kinbody.GetTransform()
        R = T[0:3,0:3]
        p = T[0:3,3:4]
        
        if kinbody.GetLinks() == 0:
            continue
        
        if len(kinbody.GetLinks()) > 1:
            print 'exporting system %s ...\n'%kinbody.GetName()
            f.write('<system name = "%s">\n'%kinbody.GetName())
        else:
            if kinbody.GetLinks()[0].IsStatic():
                ground_link.append(kinbody.GetLinks()[0])
                ground_link_T.append(T)
                continue
            print 'exporting object %s ...\n'%kinbody.GetName()
            f.write('<object name = "%s">\n'%kinbody.GetName())
        
        f.write('  <translation>%f %f %f</translation>\n'%(p[0],p[1],p[2]))
        f.write('  <rotation>%f %f %f %f %f %f %f %f %f</rotation>\n'%(R[0,0],R[1,0],R[2,0],R[0,1],R[1,1],R[2,1],R[0,2],R[1,2],R[2,2]))
        print ' exporting links...'
        for link in kinbody.GetLinks():
            print '  '+link.GetName()
            trimesh = link.GetCollisionData()
            f.write('  <body name = "%s">\n'%link.GetName())
            f.write('    <geom type = "mesh" format = "data">\n')
            f.write('      <!-- number of vertices = %d -->\n'%len(trimesh.vertices))
            f.write('      <vertices>')
            for vertex in trimesh.vertices:
                f.write('%f %f %f '%(vertex[0],vertex[1],vertex[2]))
            f.write('</vertices>\n')
            f.write('      <!-- number of faces = %d -->\n'%len(trimesh.indices))
            f.write('      <faces>')
            for idx in trimesh.indices:
                f.write('%d %d %d '%(idx[0],idx[1],idx[2]))
            f.write('</faces>\n') 
            f.write('    </geom>\n')   
            f.write('  </body>\n')
        print ' exporting (active) joints...'    
        for joint in kinbody.GetJoints():
            print '  '+joint.GetName()
            axis = joint.GetInternalHierarchyAxis(0)
            body1 = joint.GetFirstAttached()
            body2 = joint.GetSecondAttached()
            T1 = joint.GetInternalHierarchyLeftTransform()
            T2 = joint.GetInternalHierarchyRightTransform()
            R1 = T1[0:3,0:3]
            p1 = T1[0:3,3:4]
            R2 = T2[0:3,0:3]
            p2 = T2[0:3,3:4]
            iR2 = transpose(R2) # Inv(T) = (iR2,ip2) = Inv(R2,p2) = (R2^t, -R2^t * p2)
            ip2 = -dot(iR2,p2)
            f.write('  <joint name = "%s">\n'%joint.GetName())
            f.write('    <type>%s</type>\n'%joint.GetType())
            f.write('    <axis>%f %f %f</axis>\n'%(axis[0],axis[1],axis[2]))
            f.write('    <connection>\n')
            f.write('      <body>%s</body>\n'%body1.GetName())
            f.write('      <translation>%f %f %f</translation>\n'%(p1[0],p1[1],p1[2]))
            f.write('      <rotation>%f %f %f %f %f %f %f %f %f</rotation>\n'%(R1[0,0],R1[1,0],R1[2,0],R1[0,1],R1[1,1],R1[2,1],R1[0,2],R1[1,2],R1[2,2]))
            f.write('    </connection>\n')
            f.write('    <connection>\n')
            f.write('      <body>%s</body>\n'%body2.GetName())
            f.write('      <translation>%f %f %f</translation>\n'%(ip2[0],ip2[1],ip2[2]))
            f.write('      <rotation>%f %f %f %f %f %f %f %f %f</rotation>\n'%(iR2[0,0],iR2[1,0],iR2[2,0],iR2[0,1],iR2[1,1],iR2[2,1],iR2[0,2],iR2[1,2],iR2[2,2]))
            f.write('    </connection>\n')
            f.write('  </joint>\n')    
        
        print ' exporting (passive) joints...'    
        for joint in kinbody.GetPassiveJoints():
            print '  '+joint.GetName()
            axis = joint.GetInternalHierarchyAxis(0)
            body1 = joint.GetFirstAttached()
            body2 = joint.GetSecondAttached()
            T1 = joint.GetInternalHierarchyLeftTransform()
            T2 = joint.GetInternalHierarchyRightTransform()
            R1 = T1[0:3,0:3]
            p1 = T1[0:3,3:4]
            R2 = T2[0:3,0:3]
            p2 = T2[0:3,3:4]
            iR2 = transpose(R2) # Inv(T) = (iR2,ip2) = Inv(R2,p2) = (R2^t, -R2^t * p2)
            ip2 = -dot(iR2,p2)
            f.write('  <joint name = "%s">\n'%joint.GetName())
            f.write('    <type>%s</type>\n'%joint.GetType())
            f.write('    <axis>%f %f %f</axis>\n'%(axis[0],axis[1],axis[2]))
            f.write('    <connection>\n')
            f.write('      <body>%s</body>\n'%body1.GetName())
            f.write('      <translation>%f %f %f</translation>\n'%(p1[0],p1[1],p1[2]))
            f.write('      <rotation>%f %f %f %f %f %f %f %f %f</rotation>\n'%(R1[0,0],R1[1,0],R1[2,0],R1[0,1],R1[1,1],R1[2,1],R1[0,2],R1[1,2],R1[2,2]))
            f.write('    </connection>\n')
            f.write('    <connection>\n')
            f.write('      <body>%s</body>\n'%body2.GetName())
            f.write('      <translation>%f %f %f</translation>\n'%(ip2[0],ip2[1],ip2[2]))
            f.write('      <rotation>%f %f %f %f %f %f %f %f %f</rotation>\n'%(iR2[0,0],iR2[1,0],iR2[2,0],iR2[0,1],iR2[1,1],iR2[2,1],iR2[0,2],iR2[1,2],iR2[2,2]))
            f.write('    </connection>\n')
            f.write('  </joint>\n')    
                
        if len(kinbody.GetLinks()) > 1:
            f.write('</system>\n\n')
        else:
            f.write('</object>\n\n')
            
    # ground
    f.write('<ground name = "ground">\n')
    for i, link in enumerate(ground_link):
        R = ground_link_T[i][0:3,0:3]
        p = ground_link_T[i][0:3,3:4]
        trimesh = link.GetCollisionData()
        f.write('    <geom type = "mesh" format = "data">\n')
        f.write('      <translation>%f %f %f</translation>\n'%(p[0],p[1],p[2]))
        f.write('      <rotation>%f %f %f %f %f %f %f %f %f</rotation>\n'%(R[0,0],R[1,0],R[2,0],R[0,1],R[1,1],R[2,1],R[0,2],R[1,2],R[2,2]))
        f.write('      <!-- number of vertices = %d -->\n'%len(trimesh.vertices))
        f.write('      <vertices>')
        for vertex in trimesh.vertices:
            f.write('%f %f %f '%(vertex[0],vertex[1],vertex[2]))
        f.write('</vertices>\n')
        f.write('      <!-- number of faces = %d -->\n'%len(trimesh.indices))
        f.write('      <faces>')
        for idx in trimesh.indices:
            f.write('%d %d %d '%(idx[0],idx[1],idx[2]))
        f.write('</faces>\n') 
        f.write('    </geom>\n')   
    f.write('</ground>\n\n')
    
    f.write('</world>\n')
    
    f.close()
    
    print 'saved to %s ...\n'%options.outputfilename
    print 'press enter to exit\n'
    raw_input()    
                                
if __name__ == "__main__":
    run()

