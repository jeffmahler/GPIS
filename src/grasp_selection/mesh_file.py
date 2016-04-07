import os

import obj_file
import stp_file
import json

class MeshFile:
	@staticmethod
	def extract_mesh(filename, obj_filename, script_to_apply):
                if script_to_apply is None:
                        meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' %(filename, obj_filename)
                else:
                        meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\" -s \"%s\"' %(filename, obj_filename, script_to_apply) 
		os.system(meshlabserver_cmd)
		print 'MeshlabServer Command:', meshlabserver_cmd

		if not os.path.exists(obj_filename):
			print 'Meshlab conversion failed for', obj_filename
			return

		of = obj_file.ObjFile(obj_filename)
		return of.read()

	@staticmethod
	def write_obj(mesh, target_filename):
		target_obj_filename = target_filename + '.obj'

		oof = obj_file.ObjFile(target_obj_filename)
		oof.write(mesh)
		os.system('chmod a+rwx \"%s\"' %(target_obj_filename))

	@staticmethod
	def write_sdf(mesh, target_filename, dim, padding):
		target_obj_filename = target_filename + '.obj'
		target_sdf_filename = target_filename + '.sdf'

		sdfgen_cmd = '/home/jmahler/Libraries/SDFGen/bin/SDFGen \"%s\" %d %d' %(target_obj_filename, dim, padding)
		os.system(sdfgen_cmd)
		print 'SDF Command', sdfgen_cmd

		if not os.path.exists(target_sdf_filename):
			print 'SDF computation failed for', target_sdf_filename
			return
		os.system('chmod a+rwx \"%s\"' %(target_sdf_filename) )

	@staticmethod
	def write_stp(mesh, target_filename, min_prob = 0.05):
		target_stp_filename = target_filename + '.stp'

		stpf = stp_file.StablePoseFile()
		stpf.write_mesh_stable_poses(mesh, target_stp_filename, min_prob=min_prob)

	@staticmethod
	def write_json(mesh, target_filename):
		target_json_filename = target_filename + '.meta'

		json.dump(mesh.create_json_metadata(), open(target_json_filename, 'w'))

	@staticmethod
	def write_shot(mesh, target_filename):
		target_obj_filename = target_filename + '.obj'
		target_shot_filename = target_filename + '.ftr'

		shot_os_call = 'bin/shot_extractor %s %s' %(target_obj_filename, target_shot_filename)
		os.system(shot_os_call)

