#sys
import sys

#utils
import os.path
import json
import glob
import time
import cPickle as pickle

#OpenGL 
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

#Qt5
from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt ,QMutex, pyqtSlot
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QOpenGLWidget, QSlider,
							 QWidget)
					
#math
import numpy as np
import numpy.linalg
import math
import sklearn.preprocessing


class domeGLWidget(QOpenGLWidget):
	#signal
	xRotationChanged = pyqtSignal(int)
	yRotationChanged = pyqtSignal(int)
	zRotationChanged = pyqtSignal(int)
	xTransChanged = pyqtSignal(int)
	yTransChanged = pyqtSignal(int)
	zTransChanged = pyqtSignal(int)

	def __init__(self, parent=None):
		super(domeGLWidget, self).__init__(parent)

		#default free view
		self.x_rot = 30
		self.y_rot = 0
		self.z_rot = 0
		self.z_trans = 300
		self.x_trans = 0
		self.y_trans = 0

		#render set
		self.z_near = 0.01
		self.z_far = 5000.
		self.g_ambient = (0.35, 0.35, 0.35, 1.0)
		self.g_diffuse = (0.75, 0.75, 0.75, 0.7)
		self.g_specular = (1.0, 1.0, 1.0, 1.0)

		self.frame_id = 0

		self.lastPos = QPoint()
		self.cam_list = []
		self.cam_id = -1

		self.body_list = []
		self.hand_list = []
		self.face_list = []
		self.mesh_list = []
		
		self.vt_buffer = None
		self.vn_buffer = None
		self.vc_buffer = None
		self.inds_buffer = None
		
		self.num_faces = 0

		self.element_status = {'floor':True,'cameras':False}
		self.render_lock = QMutex()



		self.coco25_parents = [1,1,1,2,3,1,5,6,1,8,9,10,8,12,13,0,0,15,16,14,19,14,11,22,11]
		self.coco_inds = range(25)

		self.rhand_ids = range(25,45)
		self.rhand_parents = [4,1+24,2+24,3+24,4,5+24,6+24,7+24,4,9+24,10+24,11+24,4,13+24,14+24,15+24,4,17+24,18+24,19+24]
		
		self.lhand_ids = range(45,65)
		self.lhand_parents =[7,1+44,2+44,3+44,7,5+44,6+44,7+44,7,9+44,10+44,11+44,7,13+44,14+44,15+44,7,17+44,18+44,19+44]

	def getGLInfo(self):
		info = """
			Vendor: {}
			Renderer: {}
			OpenGL Version: {}
			Shader Version: {}
		""".format(
			glGetString(GL_VENDOR),
			glGetString(GL_RENDERER),
			glGetString(GL_VERSION),
			glGetString(GL_SHADING_LANGUAGE_VERSION)
		)
		return info

	def initializeGL(self):
		print(self.getGLInfo())

		glutInit(sys.argv)
		glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
		glClearColor(1.0, 1.0, 1.0, 1.0)
		glShadeModel(GL_SMOOTH)
		glLightfv(GL_LIGHT0, GL_AMBIENT, self.g_ambient)
		glLightfv(GL_LIGHT0, GL_DIFFUSE, self.g_diffuse)
		glLightfv(GL_LIGHT0, GL_SPECULAR, self.g_specular)
		glLightfv(GL_LIGHT0, GL_POSITION, [-1, -1, 0])

		glEnable(GL_LIGHTING)
		glEnable(GL_LIGHT0)
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_COLOR_MATERIAL)
		glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)

		self.vt_buffer = glGenBuffers(1)
		self.vc_buffer = glGenBuffers(1)
		self.vn_buffer = glGenBuffers(1)
		self.inds_buffer = glGenBuffers(1)
	
	#
	# def setClearColor(self, c):
	# 	glClearColor(c.redF(), c.greenF(), c.blueF(), c.alphaF())

	# def setColor(self, c):
	# 	glColor4f(c.redF(), c.greenF(), c.blueF(), c.alphaF())



	def render_mesh(self,mode=GL_FILL):
		glEnable(GL_CULL_FACE)
		glPushMatrix()
		glLineWidth(.5)
		glEnableVertexAttribArray(0)
		glEnableVertexAttribArray(3)
		glBindBuffer(GL_ARRAY_BUFFER, self.vt_buffer)
		glVertexAttribPointer(
					0,                                # attribute
					3,                                # size
					GL_FLOAT,                         # type
					GL_FALSE,                         # normalized?
					0,  # / stride
					None                         # array buffer offset
				)
		glBindBuffer(GL_ARRAY_BUFFER, self.vc_buffer)
		glVertexAttribPointer(
					3,                                # 1 attribute
					3,                                # size
					GL_FLOAT,                         # type
					GL_FALSE,                         # normalized?
					0,  # / stride
					None                         # array buffer offset
				)

		glEnableVertexAttribArray(2)
		glBindBuffer(GL_ARRAY_BUFFER, self.vn_buffer)
		glVertexAttribPointer(
					2,                                # attribute
					3,                                # size
					GL_FLOAT,                         # type
					GL_TRUE,                         # normalized?
					0,  # / stride
					None                         # array buffer offset
				)
		
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.inds_buffer)
		glPolygonMode(GL_FRONT_AND_BACK, mode)
		glDrawElements(GL_TRIANGLES, self.num_faces*3, GL_UNSIGNED_INT, None)

		glDisableVertexAttribArray(2)
		glDisableVertexAttribArray(0)
		glDisableVertexAttribArray(3)
		glPopMatrix()
	

	def set_render_content(self,body_list,mesh_list):
		self.render_lock.lock()
		self.body_list = body_list
		self.mesh_list = mesh_list

		if mesh_list is not None:

			vt = self.mesh_list['v']
			vc = self.mesh_list['vc']
			vn = self.mesh_list['vn']
			f = self.mesh_list['f']
			self.num_faces = f.shape[0]
			vt_c = vt.flatten().tolist()
			vc_c = vc.flatten().tolist()
			vn_c = vn.flatten().tolist()
			f_c = f.flatten().astype(np.int).tolist()

			glBindBuffer(GL_ARRAY_BUFFER, self.vt_buffer)
			glBufferData(GL_ARRAY_BUFFER, len(vt_c)*sizeof(ctypes.c_float),
							(ctypes.c_float*len(vt_c))(*vt_c), GL_STATIC_DRAW)

			glBindBuffer(GL_ARRAY_BUFFER, self.vc_buffer)
			glBufferData(GL_ARRAY_BUFFER, len(vc_c)*sizeof(ctypes.c_float),
							(ctypes.c_float*len(vc_c))(*vc_c), GL_STATIC_DRAW)

			glBindBuffer(GL_ARRAY_BUFFER, self.vn_buffer)
			glBufferData(GL_ARRAY_BUFFER, len(vn_c)*sizeof(ctypes.c_float),
							(ctypes.c_float*len(vn_c))(*vn_c), GL_STATIC_DRAW)

			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.inds_buffer)
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ctypes.c_int) * len(f_c),
							(ctypes.c_int * len(f_c))(*f_c), GL_STATIC_DRAW)

		self.render_lock.unlock()
		self.update()


	def set_cameras(self,cam_list):
		self.cam_list = cam_list
		self.update()

	def render_floor(self,grid_num=10,grid_size=50):

		glDisable(GL_LIGHTING)
		floor_c = np.array([0,0,0])
		floor_xdir = np.array([1,0,0])
		floor_ydir = np.array([0,0,1])

		origin = floor_c - floor_xdir*(grid_size*grid_num/2 ) - floor_ydir*(grid_size*grid_num/2)
		axis_x =  floor_xdir * grid_size
		axis_y =  floor_ydir * grid_size

		for y in range(grid_num+1):
			for x in range(grid_num+1):
				if (x+y) % 2 ==0:
					glColor(1.0,1.0,1.0,1.0) #white
				else:
					glColor(0.7,0.7,0.7,1) #grey
				p1 = origin + axis_x*x + axis_y*y
				p2 = p1+ axis_x
				p3 = p1+ axis_y
				p4 = p1+ axis_x + axis_y
				glBegin(GL_QUADS)
				glVertex3f(   p1[0], p1[1], p1[2])
				glVertex3f(   p2[0], p2[1], p2[2])
				glVertex3f(   p4[0], p4[1], p4[2])
				glVertex3f(   p3[0], p3[1], p3[2])
				glEnd()

		glEnable(GL_LIGHTING)

	def render_cameras(self):
		glDisable(GL_LIGHTING)
		sz = 10
		for cam in self.cam_list:
			invR = np.array(cam['R'])
			invT = np.array(cam['t'])
			camMatrix = np.hstack((invR,invT))
			camMatrix = np.vstack((camMatrix,[0,0,0,1]))
			camMatrix = numpy.linalg.inv(camMatrix)
			K = np.array(cam['K'])
			fx = K[0,0]
			fy = K[1,1]
			cx = K[0,2]
			cy = K[1,2]
			width = cam['resolution'][0]
			height = cam['resolution'][1]
			glPushMatrix()
			glMultMatrixf(camMatrix.T)
			glLineWidth(2)
			glColor3f(1,0,0)
			glBegin(GL_LINES)
			glVertex3f(0,0,0)
			glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz)
			glVertex3f(0,0,0)
			glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz)
			glVertex3f(0,0,0)
			glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz)
			glVertex3f(0,0,0)
			glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz)
			glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz)
			glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz)
			glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz)
			glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz)
			glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz)
			glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz)
			glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz)
			glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz)
			glEnd()
			glPopMatrix()
		glEnable(GL_LIGHTING)


	def render_joints(self):
		for entry in self.body_list:
			Jtr = np.zeros((65,4))
			Jtr[0:25,:] =  entry['joints25']
			Jtr[25:45,:] = entry['right_hand'][1:,:]
			Jtr[45:,:] = entry['left_hand'][1:,:]
			for cid,pid in zip(self.coco_inds,self.coco25_parents):
				if cid==pid:
					continue	
				if Jtr[cid,3]>0.1 and Jtr[pid,3]>0.1:
					self.render_cone(Jtr[pid,0:3],Jtr[cid,0:3],15,[0.75,0.75,0.75,1.0])
			for cid,pid in zip(self.rhand_ids,self.rhand_parents):
				if cid==pid:
					continue	
				if Jtr[cid,3]>0.1 and Jtr[pid,3]>0.1:
					self.render_cone(Jtr[pid,0:3],Jtr[cid,0:3],5,[0.80,0.80,0.80,1.0])

			for cid,pid in zip(self.lhand_ids,self.lhand_parents):
				if cid==pid:
					continue
				if Jtr[cid,3]>0.1 and Jtr[pid,3]>0.1:
					self.render_cone(Jtr[pid,0:3],Jtr[cid,0:3],5,[0.80,0.80,0.80,1.0])

	def render_cone(self,v1,v2,dim,color):
		glColor4f(color[0], color[1], color[2], color[3])
		v2r = v2 - v1
		z = np.array([0.0, 0.0, 1.0])
		# the rotation axis is the cross product between Z and v2r
		ax = np.cross(z, v2r)
		import math
		l = math.sqrt(np.dot(v2r, v2r))
		# get the angle using a dot product
		angle = 180.0 / math.pi * math.acos(np.dot(z, v2r) / l)

		glPushMatrix()
		glTranslatef(v1[0], v1[1], v1[2])
		#print "The cylinder between %s and %s has angle %f and axis %s\n" % (v1, v2, angle, ax)
		glRotatef(angle, ax[0], ax[1], ax[2])
		glutSolidCone(dim / 10.0, l, 20, 20)
		glPopMatrix()




	

	# paintGL is the main loop
	def paintGL(self):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

		self.lookAt()
		#
		self.render_lock.lock()
		for elem,status in self.element_status.items():
			if status:
				getattr(self,'render_{}'.format(elem))()
		self.render_joints()
		
		if self.mesh_list is not None:
			self.render_mesh()
		self.render_lock.unlock()

	def resizeGL(self, width, height):
		glViewport(0, 0, width, height)



	#event and signal handler
	def setXRotation(self, angle):
		angle = self.normalizeAngle(angle)
		if angle != self.x_rot:
			self.x_rot = angle
			self.xRotationChanged.emit(angle)
			self.update()

	def setYRotation(self, angle):
		angle = self.normalizeAngle(angle)
		if angle != self.y_rot:
			self.y_rot = angle
			self.yRotationChanged.emit(angle)
			self.update()

	def setZRotation(self, angle):
		angle = self.normalizeAngle(angle)
		if angle != self.z_rot:
			self.z_rot = angle
			self.zRotationChanged.emit(angle)
			self.update()
	def setXTrans(self,trans):
		if trans !=self.x_trans:
			self.x_trans = trans
			self.xTransChanged.emit(trans)
			self.update()
	def setYTrans(self,trans):
		if trans !=self.y_trans:
			self.y_trans = trans
			self.yTransChanged.emit(trans)
			self.update()
	def setZTrans(self,trans):
		if trans !=self.z_trans:
			self.z_trans = trans
			self.zTransChanged.emit(trans)
			self.update()

	#mouse event handler
	def mousePressEvent(self, event):
		self.lastPos = event.pos()

	def mouseMoveEvent(self, event):
		dx = event.x() - self.lastPos.x()
		dy = event.y() - self.lastPos.y()
		
		if event.buttons() & Qt.LeftButton:
			self.setXRotation(self.x_rot + 2 * dy)
			self.setYRotation(self.y_rot + 2 * dx)
		#temporally disable translation
		# elif event.buttons() & Qt.RightButton:
		# 	self.setXTrans(self.x_trans+8*dx)
		# 	self.setYTrans(self.y_trans+8*dy)
		self.lastPos = event.pos()

	def wheelEvent(self, event):
		dz = event.angleDelta().y()/8
		self.setZTrans(self.z_trans+dz)

	def normalizeAngle(self, angle):
		while angle < 0:
			angle += 360
		while angle > 360:
			angle -= 360
		return angle



	
	

	def lookAt(self):
		if self.cam_id < 0:
			#projection
			glMatrixMode(GL_PROJECTION)
			glLoadIdentity()
			gluPerspective(65, float(self.width())/float(self.height()), self.z_near, self.z_far)

			#model view
			glMatrixMode(GL_MODELVIEW)
			glLoadIdentity()
			gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)

			glTranslatef(0, 0, self.z_trans)
			glRotatef(self.x_rot, 1.0, 0.0, 0.0)
			glRotatef(self.y_rot, 0.0, 1.0, 0.0)
			glRotatef(self.z_rot, 0.0, 0.0, 1.0)
			glTranslatef(self.x_trans,  0.0, 0.0)
			glTranslatef(0.0, self.y_trans, 0.0)
		else:

			cam = self.cam_list[self.cam_id]
			invR = np.array(cam['R'])
			invT = np.array(cam['t'])
			camMatrix = np.hstack((invR, invT))
			# denotes camera matrix, [R|t]
			camMatrix = np.vstack((camMatrix, [0, 0, 0, 1]))
			#camMatrix = numpy.linalg.inv(camMatrix)
			K = np.array(cam['K'])
			#K = K.flatten()
			glMatrixMode(GL_PROJECTION)
			glLoadIdentity()
			Kscale = cam['resolution'][0]*1.0/self.width()
			K = K/Kscale
			ProjM = np.zeros((4,4))
			ProjM[0,0] = 2*K[0,0]/self.width()
			ProjM[0,2] = (self.width() - 2*K[0,2])/self.width()
			ProjM[1,1] = 2*K[1,1]/self.height()
			ProjM[1,2] = (-self.height()+2*K[1,2])/self.height()
			ProjM[2,2] = (-self.z_far-self.z_near)/(self.z_far-self.z_near)
			ProjM[2,3] = -2*self.z_far*self.z_near/(self.z_far-self.z_near)
			ProjM[3,2] = -1

			glLoadMatrixd(ProjM.T)
			glMatrixMode(GL_MODELVIEW)
			glLoadIdentity()
			gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0)
			glMultMatrixd(camMatrix.T)


	@pyqtSlot(int)
	def setCameraStatus(self,camIndex):
		self.cam_id = camIndex - 1
		self.update()
	def clear_smpls(self):
		self.render_lock.lock()
		self.vt_buffer = None
		self.vc_buffer = None
		self.vn_buffer = None
		self.render_lock.unlock()


class Window(QWidget):

	def __init__(self):
		super(Window, self).__init__()

		self.glWidget = domeGLWidget()

		self.xSlider = self.createSlider()
		self.ySlider = self.createSlider()
		self.zSlider = self.createSlider()

		self.xSlider.valueChanged.connect(self.glWidget.setXRotation)
		self.glWidget.xRotationChanged.connect(self.xSlider.setValue)
		self.ySlider.valueChanged.connect(self.glWidget.setYRotation)
		self.glWidget.yRotationChanged.connect(self.ySlider.setValue)
		self.zSlider.valueChanged.connect(self.glWidget.setZRotation)
		self.glWidget.zRotationChanged.connect(self.zSlider.setValue)

		mainLayout = QHBoxLayout()
		mainLayout.addWidget(self.glWidget)
		mainLayout.addWidget(self.xSlider)
		mainLayout.addWidget(self.ySlider)
		mainLayout.addWidget(self.zSlider)
		self.setLayout(mainLayout)

		self.xSlider.setValue(15 * 16)
		self.ySlider.setValue(345 * 16)
		self.zSlider.setValue(0 * 16)

		self.setWindowTitle("Panoptic Studio Viewer")

	def createSlider(self):
		slider = QSlider(Qt.Vertical)

		slider.setRange(0, 360 * 16)
		slider.setSingleStep(16)
		slider.setPageStep(15 * 16)
		slider.setTickInterval(15 * 16)
		slider.setTickPosition(QSlider.TicksRight)

		return slider


if __name__ == '__main__':

	app = QApplication(sys.argv)
	window = Window()
	window.show()
	sys.exit(app.exec_())
