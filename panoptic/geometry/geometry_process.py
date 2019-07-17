import scipy.sparse
import numpy as np
import sklearn.preprocessing

def vertices_normals(f,v):

	fNormal_u = v[f[:,1],:] - v[f[:,0],:]
	fNormal_v = v[f[:,2],:] - v[f[:,0],:]
	fNormal = np.cross(fNormal_u,fNormal_v)
	fNormal = sklearn.preprocessing.normalize(fNormal)
	
	vbyf_vid = f.flatten('F')
	vbyf_fid = np.arange(f.shape[0])
	vbyf_fid = np.concatenate((vbyf_fid,vbyf_fid,vbyf_fid))
	vbyf_val = np.ones(len(vbyf_vid))
	vbyf = scipy.sparse.coo_matrix((vbyf_val,(vbyf_vid,vbyf_fid)),shape=(v.shape[0],f.shape[0])).tocsr()
	
	vNormal = vbyf.dot(fNormal)

	vNormal = sklearn.preprocessing.normalize(vNormal)

	return vNormal
