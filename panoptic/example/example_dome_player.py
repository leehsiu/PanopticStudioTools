import sys
from PyQt5 import QtWidgets
from panoptic.app.pyqtDomePlayer import PanopticStudioPlayer
import os.path
import json
import cPickle as pickle
import glob
import numpy as np
import panoptic.geometry.geometry_process as geo_utils
from panoptic.bodymodel.batch_total import BatchTotalNumpy
import time
if __name__=='__main__':

	app = QtWidgets.QApplication(sys.argv)
	m_player = PanopticStudioPlayer()
	m_player.dataroot = '/media/posefs3b/Users/xiu/domedb'
	m_player.seqname = '171204_pose3'

	adamPath = '/home/xiul/workspace/FullyCapture/models/adamModel_with_coco25_reg.pkl'

	totalModelWrapper = BatchTotalNumpy(pkl_path=adamPath)

	joints_file = glob.glob(os.path.join(m_player.dataroot,m_player.seqname,'hdPose3d_stage0_coco25','*.json'))

	joints_file.sort()

	def custom_load_cameras(dataroot,seqname):
		global_root = '/media/posefs0c/panoptic'
		calib_file = os.path.join(global_root,seqname,'calibration_{}.json'.format(seqname))
		with open(calib_file) as f:
			calib_json_str = json.load(f)
		cameras = calib_json_str['cameras']
		allPanel = map(lambda x:x['panel'],cameras)
		hdCamIndices = [i for i,x in enumerate(allPanel) if x==0]
		cam_list = [cameras[i] for i in hdCamIndices]
		return cam_list


	def custom_load_timeline(dataroot,seqname):
		frame_num = len(joints_file)
		return frame_num

	def custom_load_joints(dataroot,seqname,frame_id):
		coco25_file = joints_file[frame_id]
		joints_total = []
		with open(coco25_file) as fio:
			jsonStr = json.load(fio)
		for entry  in jsonStr['bodies']:
			cid = entry['id']
			joints25 = np.reshape(entry['joints25'],(-1,4))
			c_joint = {'id':cid,
					   'joints25':joints25,
					   'right_hand':np.zeros((21,4)),
					   'left_hand':np.zeros((21,4)),
					   'face70':np.zeros((70,4))
						}
			joints_total.append(c_joint)
		return joints_total

	def custom_load_mesh(dataroot,seqname,frame_id):		
		#dataroot is '/media/posefs3b/Users/
		t0 = time.time()

		coco25_file = joints_file[frame_id]
		coco25_file_id = m_player.default_filename_to_frameid(coco25_file)

		mesh_file = os.path.join('/media/posefs3b/Users/xiu/domedb',seqname,'hdPose3d_Adam_stage1','bodyAdam_{:08d}.pkl'.format(coco25_file_id))
		
		if os.path.isfile(mesh_file):
			with open(mesh_file) as fio:
				mesh_param = pickle.load(fio)
			trans = []
			betas = []
			poses = []
			ids = []

			for entry in mesh_param:
				trans.append(entry['trans'])
				betas.append(entry['betas'])
				poses.append(entry['pose'])
				ids.append(entry['id'])
			trans = np.array(trans)[:,None,:]
			betas = np.array(betas)
			poses = np.array(poses)
			vt,_ = totalModelWrapper(betas,poses,get_skin=True)
			vt += trans
			vt = vt.reshape(-1,3)
			vf = []
			vc = []
			for ip,id_c in enumerate(ids):
				vf.append(totalModelWrapper.f + ip*totalModelWrapper.size[0])
				c_level = (id_c % 10) + 1
				c_level = c_level * 0.05
				vc.append(np.ones(totalModelWrapper.size)-c_level)
			vf = np.concatenate(vf)
			vc = np.concatenate(vc)
			vn = geo_utils.vertices_normals(f = vf,v=vt)
		
			t1 = time.time()
			print('time for create mesh {}'.format(t1-t0))
			return {'v':vt,'f':vf,'vc':vc,'vn':vn}
		else:
			return None


	m_player.custom_load_cameras = custom_load_cameras
	m_player.custom_load_timeline = custom_load_timeline
	m_player.custom_load_joints = custom_load_joints
	m_player.custom_load_mesh = custom_load_mesh
	m_player.reset_widget()
	m_player.loadDB_core()
	m_player.show()

	sys.exit(app.exec_())  

