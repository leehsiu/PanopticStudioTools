import sys

from PyQt5.QtCore import pyqtSlot,pyqtSignal
from panoptic.gl.pyqtGLWidget import domeGLWidget
from PyQt5 import QtCore,QtWidgets,QtGui
import json
import os.path
import glob
import numpy as np
import copy

class Ui_MainWindow(object):
	def setupUi(self, MainWindow):
		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(1184, 706)
		self.centralwidget = QtWidgets.QWidget(MainWindow)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
		self.centralwidget.setSizePolicy(sizePolicy)
		self.centralwidget.setMinimumSize(QtCore.QSize(0, 0))
		self.centralwidget.setObjectName("centralwidget")
		self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
		self.verticalLayout.setObjectName("verticalLayout")
		self.mainLayout = QtWidgets.QHBoxLayout()
		self.mainLayout.setObjectName("mainLayout")
		self.widget_player = QtWidgets.QWidget(self.centralwidget)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.widget_player.sizePolicy().hasHeightForWidth())
		self.widget_player.setSizePolicy(sizePolicy)
		self.widget_player.setObjectName("widget_player")
		self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget_player)
		self.verticalLayout_2.setObjectName("verticalLayout_2")

		self.mainGlWidget = domeGLWidget(self.widget_player)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.mainGlWidget.sizePolicy().hasHeightForWidth())
		self.mainGlWidget.setSizePolicy(sizePolicy)
		self.mainGlWidget.setMinimumSize(QtCore.QSize(0, 0))
		self.mainGlWidget.setObjectName("mainGlWidget")
		self.verticalLayout_2.addWidget(self.mainGlWidget)

		self.widget_timeline = QtWidgets.QWidget(self.widget_player)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.widget_timeline.sizePolicy().hasHeightForWidth())
		self.widget_timeline.setSizePolicy(sizePolicy)
		self.widget_timeline.setObjectName("widget_timeline")
		self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_timeline)
		self.horizontalLayout.setObjectName("horizontalLayout")
		self.frame_slider = QtWidgets.QSlider(self.widget_timeline)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.frame_slider.sizePolicy().hasHeightForWidth())
		self.frame_slider.setSizePolicy(sizePolicy)
		self.frame_slider.setOrientation(QtCore.Qt.Horizontal)
		self.frame_slider.setObjectName("frame_slider")
		self.horizontalLayout.addWidget(self.frame_slider)
		self.bn_slow = QtWidgets.QToolButton(self.widget_timeline)
		self.bn_slow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
		self.bn_slow.setArrowType(QtCore.Qt.NoArrow)
		self.bn_slow.setObjectName("bn_slow")
		self.horizontalLayout.addWidget(self.bn_slow)
		self.bn_play = QtWidgets.QToolButton(self.widget_timeline)
		self.bn_play.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
		self.bn_play.setArrowType(QtCore.Qt.NoArrow)
		self.bn_play.setObjectName("bn_play")
		self.horizontalLayout.addWidget(self.bn_play)
		self.bn_fast = QtWidgets.QToolButton(self.widget_timeline)
		self.bn_fast.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
		self.bn_fast.setArrowType(QtCore.Qt.NoArrow)
		self.bn_fast.setObjectName("bn_fast")
		self.horizontalLayout.addWidget(self.bn_fast)
		self.edit_cur_frame = QtWidgets.QLineEdit(self.widget_timeline)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.edit_cur_frame.sizePolicy().hasHeightForWidth())
		self.edit_cur_frame.setSizePolicy(sizePolicy)
		self.edit_cur_frame.setMaximumSize(QtCore.QSize(100, 30))
		self.edit_cur_frame.setAutoFillBackground(False)
		self.edit_cur_frame.setMaxLength(8)
		self.edit_cur_frame.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
		self.edit_cur_frame.setObjectName("edit_cur_frame")
		self.horizontalLayout.addWidget(self.edit_cur_frame)
		self.label_frame = QtWidgets.QLabel(self.widget_timeline)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.label_frame.sizePolicy().hasHeightForWidth())
		self.label_frame.setSizePolicy(sizePolicy)
		self.label_frame.setTextFormat(QtCore.Qt.AutoText)
		self.label_frame.setObjectName("label_frame")
		self.horizontalLayout.addWidget(self.label_frame)
		self.edit_max_frame = QtWidgets.QLineEdit(self.widget_timeline)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.edit_max_frame.sizePolicy().hasHeightForWidth())
		self.edit_max_frame.setSizePolicy(sizePolicy)
		self.edit_max_frame.setMaximumSize(QtCore.QSize(100, 30))
		self.edit_max_frame.setReadOnly(True)
		self.edit_max_frame.setClearButtonEnabled(False)
		self.edit_max_frame.setObjectName("edit_max_frame")
		self.horizontalLayout.addWidget(self.edit_max_frame)
		self.verticalLayout_2.addWidget(self.widget_timeline)
		self.mainLayout.addWidget(self.widget_player)
		self.widget_split = QtWidgets.QWidget(self.centralwidget)
		self.widget_split.setObjectName("widget_split")
		self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget_split)
		self.verticalLayout_3.setObjectName("verticalLayout_3")
		self.bn_showcontrol = QtWidgets.QToolButton(self.widget_split)
		self.bn_showcontrol.setCheckable(True)
		self.bn_showcontrol.setChecked(True)
		self.bn_showcontrol.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
		self.bn_showcontrol.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
		self.bn_showcontrol.setArrowType(QtCore.Qt.RightArrow)
		self.bn_showcontrol.setObjectName("bn_showcontrol")
		self.verticalLayout_3.addWidget(self.bn_showcontrol)
		spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
		self.verticalLayout_3.addItem(spacerItem)
		self.mainLayout.addWidget(self.widget_split)
		self.widget_control = QtWidgets.QWidget(self.centralwidget)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.widget_control.sizePolicy().hasHeightForWidth())
		self.widget_control.setSizePolicy(sizePolicy)
		self.widget_control.setObjectName("widget_control")
		self.layout_control = QtWidgets.QVBoxLayout(self.widget_control)
		self.layout_control.setObjectName("layout_control")
		self.group_model = QtWidgets.QGroupBox(self.widget_control)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.group_model.sizePolicy().hasHeightForWidth())
		self.group_model.setSizePolicy(sizePolicy)
		self.group_model.setFlat(False)
		self.group_model.setObjectName("group_model")
		self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.group_model)
		self.verticalLayout_4.setObjectName("verticalLayout_4")
		self.list_model = QtWidgets.QTreeWidget(self.group_model)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.list_model.sizePolicy().hasHeightForWidth())
		self.list_model.setSizePolicy(sizePolicy)
		self.list_model.setObjectName("list_model")
		self.list_model.headerItem().setText(0, "1")
		self.verticalLayout_4.addWidget(self.list_model)
		self.layout_control.addWidget(self.group_model)
		spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
		self.layout_control.addItem(spacerItem1)
		self.group_cam = QtWidgets.QGroupBox(self.widget_control)
		sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.group_cam.sizePolicy().hasHeightForWidth())
		self.group_cam.setSizePolicy(sizePolicy)
		self.group_cam.setObjectName("group_cam")
		self.gridLayout = QtWidgets.QGridLayout(self.group_cam)
		self.gridLayout.setObjectName("gridLayout")
		self.label_y = QtWidgets.QLabel(self.group_cam)
		self.label_y.setObjectName("label_y")
		self.gridLayout.addWidget(self.label_y, 5, 1, 1, 1)
		self.label_z = QtWidgets.QLabel(self.group_cam)
		self.label_z.setObjectName("label_z")
		self.gridLayout.addWidget(self.label_z, 5, 2, 1, 1)
		self.label_xrot = QtWidgets.QLabel(self.group_cam)
		self.label_xrot.setObjectName("label_xrot")
		self.gridLayout.addWidget(self.label_xrot, 2, 0, 1, 1)
		self.xEdit = QtWidgets.QDoubleSpinBox(self.group_cam)
		self.xEdit.setMinimum(-200.0)
		self.xEdit.setMaximum(200.0)
		self.xEdit.setObjectName("xEdit")
		self.gridLayout.addWidget(self.xEdit, 6, 0, 1, 1)
		self.zRotEdit = QtWidgets.QDoubleSpinBox(self.group_cam)
		self.zRotEdit.setMaximum(360.0)
		self.zRotEdit.setObjectName("zRotEdit")
		self.gridLayout.addWidget(self.zRotEdit, 4, 2, 1, 1)
		self.zEdit = QtWidgets.QDoubleSpinBox(self.group_cam)
		self.zEdit.setMinimum(-200.0)
		self.zEdit.setMaximum(200.0)
		self.zEdit.setObjectName("zEdit")
		self.gridLayout.addWidget(self.zEdit, 6, 2, 1, 1)
		self.yRotEdit = QtWidgets.QDoubleSpinBox(self.group_cam)
		self.yRotEdit.setMaximum(360.0)
		self.yRotEdit.setObjectName("yRotEdit")
		self.gridLayout.addWidget(self.yRotEdit, 4, 1, 1, 1)
		self.label_yrot = QtWidgets.QLabel(self.group_cam)
		self.label_yrot.setObjectName("label_yrot")
		self.gridLayout.addWidget(self.label_yrot, 2, 1, 1, 1)
		self.yEdit = QtWidgets.QDoubleSpinBox(self.group_cam)
		self.yEdit.setMinimum(-200.0)
		self.yEdit.setMaximum(200.0)
		self.yEdit.setObjectName("yEdit")
		self.gridLayout.addWidget(self.yEdit, 6, 1, 1, 1)
		self.xRotEdit = QtWidgets.QDoubleSpinBox(self.group_cam)
		self.xRotEdit.setMinimum(0.0)
		self.xRotEdit.setMaximum(360.0)
		self.xRotEdit.setObjectName("xRotEdit")
		self.gridLayout.addWidget(self.xRotEdit, 4, 0, 1, 1)
		self.label_x = QtWidgets.QLabel(self.group_cam)
		self.label_x.setObjectName("label_x")
		self.gridLayout.addWidget(self.label_x, 5, 0, 1, 1)
		self.label_zrot = QtWidgets.QLabel(self.group_cam)
		self.label_zrot.setObjectName("label_zrot")
		self.gridLayout.addWidget(self.label_zrot, 2, 2, 1, 1)
		self.bn_reset_cam = QtWidgets.QPushButton(self.group_cam)
		self.bn_reset_cam.setObjectName("bn_reset_cam")
		self.gridLayout.addWidget(self.bn_reset_cam, 1, 2, 1, 1)
		self.camCombo = QtWidgets.QComboBox(self.group_cam)
		self.camCombo.setObjectName("camCombo")
		self.gridLayout.addWidget(self.camCombo, 1, 0, 1, 2)
		self.layout_control.addWidget(self.group_cam)
		self.mainLayout.addWidget(self.widget_control)
		self.verticalLayout.addLayout(self.mainLayout)
		MainWindow.setCentralWidget(self.centralwidget)
		self.menubar = QtWidgets.QMenuBar(MainWindow)
		self.menubar.setGeometry(QtCore.QRect(0, 0, 1184, 25))
		self.menubar.setObjectName("menubar")
		self.menuFiles = QtWidgets.QMenu(self.menubar)
		self.menuFiles.setObjectName("menuFiles")
		self.menuCapture = QtWidgets.QMenu(self.menubar)
		self.menuCapture.setObjectName("menuCapture")
		MainWindow.setMenuBar(self.menubar)
		self.statusbar = QtWidgets.QStatusBar(MainWindow)
		self.statusbar.setEnabled(True)
		self.statusbar.setSizeGripEnabled(True)
		self.statusbar.setObjectName("statusbar")
		MainWindow.setStatusBar(self.statusbar)
		self.actionLoad = QtWidgets.QAction(MainWindow)
		self.actionLoad.setObjectName("actionLoad")
		self.actionLoad_calibs = QtWidgets.QAction(MainWindow)
		self.actionLoad_calibs.setObjectName("actionLoad_calibs")
		self.actionRecord = QtWidgets.QAction(MainWindow)
		self.actionRecord.setObjectName("actionRecord")
		self.actionStop = QtWidgets.QAction(MainWindow)
		self.actionStop.setObjectName("actionStop")
		self.actionClear = QtWidgets.QAction(MainWindow)
		self.actionClear.setObjectName("actionClear")
		self.actionLoad_Skeletons = QtWidgets.QAction(MainWindow)
		self.actionLoad_Skeletons.setObjectName("actionLoad_Skeletons")
		self.actionConnectDB = QtWidgets.QAction(MainWindow)
		self.actionConnectDB.setObjectName("actionConnectDB")
		self.actionShow_Database_List = QtWidgets.QAction(MainWindow)
		self.actionShow_Database_List.setObjectName("actionShow_Database_List")
		self.actionLoadDB = QtWidgets.QAction(MainWindow)
		self.actionLoadDB.setObjectName("actionLoadDB")
		self.menuFiles.addAction(self.actionLoadDB)
		self.menuFiles.addAction(self.actionClear)
		self.menuCapture.addAction(self.actionRecord)
		self.menuCapture.addAction(self.actionStop)
		self.menubar.addAction(self.menuFiles.menuAction())
		self.menubar.addAction(self.menuCapture.menuAction())

		self.retranslateUi(MainWindow)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)

	def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "Panoptic Studio Viewer"))
		self.bn_slow.setText(_translate("MainWindow", "..."))
		self.bn_play.setText(_translate("MainWindow", "..."))
		self.bn_fast.setText(_translate("MainWindow", "..."))
		self.label_frame.setText(_translate("MainWindow", "/"))
		self.bn_showcontrol.setText(_translate("MainWindow", "..."))
		self.group_model.setTitle(_translate("MainWindow", "Models"))
		self.group_cam.setTitle(_translate("MainWindow", "Camera"))
		self.label_y.setText(_translate("MainWindow", "y"))
		self.label_z.setText(_translate("MainWindow", "z"))
		self.label_xrot.setText(_translate("MainWindow", "xRot"))
		self.label_yrot.setText(_translate("MainWindow", "yRot"))
		self.label_x.setText(_translate("MainWindow", "x"))
		self.label_zrot.setText(_translate("MainWindow", "zRot"))
		self.bn_reset_cam.setText(_translate("MainWindow", "Reset"))
		self.menuFiles.setTitle(_translate("MainWindow", "Files"))
		self.menuCapture.setTitle(_translate("MainWindow", "Capture"))
		self.actionLoad.setText(_translate("MainWindow", "Load Mesh"))
		self.actionLoad_calibs.setText(_translate("MainWindow", "Load calibs"))
		self.actionRecord.setText(_translate("MainWindow", "Record"))
		self.actionStop.setText(_translate("MainWindow", "Stop"))
		self.actionClear.setText(_translate("MainWindow", "Clear"))
		self.actionLoad_Skeletons.setText(_translate("MainWindow", "Load Skeletons"))
		self.actionConnectDB.setText(_translate("MainWindow", "Connect to Database"))
		self.actionShow_Database_List.setText(_translate("MainWindow", "Show Dataset List"))
		self.actionLoadDB.setText(_translate("MainWindow", "Load"))




class PanopticStudioPlayer(QtWidgets.QMainWindow,Ui_MainWindow):

	frameIdChanged = pyqtSignal(int)

	def __init__(self):
		super(PanopticStudioPlayer, self).__init__()
		self.setupUi(self)
		
		#set play button
		style = self.bn_play.style()
		self.bn_play.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaPlay))
		self.bn_slow.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaSkipBackward))
		self.bn_fast.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaSkipForward))
		self.bn_play.clicked.connect(self.bn_play_clicked)
		self.actionLoadDB.triggered.connect(self.loadDB)
	
		#self.glFrame.resizeEvent = self.glWidget_resize
		#self.mainGlWidget = domeGLWidget(self)
		#self.mainGlWidget.setGeometry(self.glFrame.geometry())
		

		self.frame_slider.setMinimum(0)
		self.frame_slider.valueChanged.connect(self.set_frameid)



		self.frameIdChanged.connect(self.frame_slider.setValue)
		self.xEdit.valueChanged.connect(self.mainGlWidget.setXTrans)
		self.mainGlWidget.xTransChanged.connect(self.xEdit.setValue)
		self.yEdit.valueChanged.connect(self.mainGlWidget.setYTrans)
		self.mainGlWidget.yTransChanged.connect(self.yEdit.setValue)
		self.zEdit.valueChanged.connect(self.mainGlWidget.setZTrans)
		self.mainGlWidget.zTransChanged.connect(self.zEdit.setValue)
		self.xRotEdit.valueChanged.connect(self.mainGlWidget.setXRotation)
		self.mainGlWidget.xRotationChanged.connect(self.xRotEdit.setValue)
		self.yRotEdit.valueChanged.connect(self.mainGlWidget.setYRotation)
		self.mainGlWidget.yRotationChanged.connect(self.yRotEdit.setValue)
		self.zRotEdit.valueChanged.connect(self.mainGlWidget.setZRotation)
		self.mainGlWidget.zRotationChanged.connect(self.zRotEdit.setValue)
		
		self.bn_showcontrol.toggled.connect(self.widget_control.setVisible)


		self.list_model.setColumnCount(1)
		self.list_model.setHeaderLabels(['Elements'])
		self.camCombo.currentIndexChanged.connect(self.mainGlWidget.setCameraStatus)


		self.reset_widget()
		
		self.timer = QtCore.QBasicTimer()
		
		self.dataroot = None
		self.seqname = None
		self.frame_id = 0
		self.frame_base = 0
		self.frame_num = 0

		self.fps = 20
		self.paused = True

		self.timer.start(1000/self.fps,self)

		
		self.custom_load_cameras = None
		self.custom_load_timeline = None
		self.custom_load_joints = None
		self.custom_load_mesh = None



	#default functions.
	def default_filename_to_frameid(self,file_name):
		base_name = os.path.basename(file_name)
		base_name = os.path.splitext(base_name)[0]
		file_id = int(base_name[-8:])
		return file_id
	
	
	
	def default_load_cameras(self):
		calib_file = os.path.join(self.dataroot,self.seqname,'calibration_{}.json'.format(self.seqname))
		with open(calib_file) as f:
			calib_json_str = json.load(f)
		cameras = calib_json_str['cameras']
		allPanel = map(lambda x:x['panel'],cameras)
		hdCamIndices = [i for i,x in enumerate(allPanel) if x==0]
		cam_list = [cameras[i] for i in hdCamIndices]
		return cam_list


	def default_load_timeline(self):
		body_files = glob.glob(os.path.join(self.dataroot,self.seqname,'hdPose3d_stage1_coco19','*.json'))
		body_files.sort()

		self.frame_base = self.default_filename_to_frameid(body_files[0])
		frame_num = self.default_filename_to_frameid(body_files[-1]) - self.frame_base + 1
		return frame_num

	def default_load_joints(self):
		joints_total = []
		coco19_id = [1,0,8,5,6,7,12,13,14,2,3,4,9,10,11,16,18,15,17] #in coco25

		frame_id = self.frame_id + self.frame_base
		body_file = os.path.join(self.dataroot,self.seqname,'hdPose3d_stage1_coco19','body3DScene_{:08d}.json'.format(frame_id))
		face_file = os.path.join(self.dataroot,self.seqname,'hdFace3d','faceRecon3D_hd{:08d}.json'.format(frame_id))
		hand_file = os.path.join(self.dataroot,self.seqname,'hdHand3d','handRecon3D_hd{:08d}.json'.format(frame_id))


		with open(body_file) as f:
			jsonDataJtr = json.load(f)
		if not jsonDataJtr['bodies']:
			return joints_total


		if os.path.isfile(hand_file):
			with open(hand_file) as f:
				jsonDataHand = json.load(f)
			handIdList = [ih['id'] for ih in jsonDataHand['people']]
		else:
			handIdList = []

		if os.path.isfile(face_file):
			with open(face_file) as f:
				jsonDataFace = json.load(f)
			faceIdList = [ih['id'] for ih in jsonDataFace['people']]
		else:
			faceIdList = []

		num_body = len(jsonDataJtr['bodies'])

		for body_id in range(num_body):
			c_id = jsonDataJtr['bodies'][body_id]['id']
			joints25 = np.zeros((25,4),dtype=float)
			
			joints19 = jsonDataJtr['bodies'][body_id]['joints19']
			joints19 = np.reshape(joints19, (-1, 4))
			joints25[coco19_id,:] = joints19

			h_id = next((i for i, x in enumerate(handIdList) if x == c_id), None)
			f_id = next((i for i, x in enumerate(faceIdList) if x == c_id), None)
			if h_id is not None and 'left_hand' in jsonDataHand['people'][h_id]:
				lHand = jsonDataHand['people'][h_id]['left_hand']['landmarks']
				lHand = np.reshape(lHand, (-1, 3))
				lhandScore = jsonDataHand['people'][h_id]['left_hand']['averageScore']
				lhandScore = np.reshape(lhandScore,(-1,1))
				lHand = np.hstack((lHand,lhandScore))
			else:
				lHand = np.zeros((21, 4))

			if h_id is not None and 'right_hand' in jsonDataHand['people'][h_id]:
				rHand = jsonDataHand['people'][h_id]['right_hand']['landmarks']
				rHand = np.reshape(rHand, (-1, 3))
				rhandScore = jsonDataHand['people'][h_id]['right_hand']['averageScore']
				rhandScore = np.reshape(rhandScore,(-1,1))
				rHand = np.hstack((rHand,rhandScore))
			else:
				rHand = np.zeros((21, 4))

			if f_id is not None and 'face70' in jsonDataFace['people'][f_id]:
				JFace = jsonDataFace['people'][f_id]['face70']['landmarks']
				JFace = np.reshape(JFace, (-1, 3))
				JFaceScore = jsonDataFace['people'][f_id]['face70']['averageScore']
				JFaceScore = np.reshape(JFaceScore,(-1,1))
				JFace = np.hstack((JFace,JFaceScore))
			else:
				JFace = np.zeros((70, 4))

			c_Total = {
						"id":c_id,
						"joints25":joints25,
						"left_hand":lHand,
						"right_hand":rHand,
						"face70":JFace
					  }
			joints_total.append(copy.deepcopy(c_Total))

		return joints_total

	def default_load_mesh(self):
		#the mesh is only stored in 
		return 0

	# def glWidget_resize(self,event):
	# 	self.mainGlWidget.resize(event.size())	
	# 	self.mainGlWidget.updateGeometry()
	def bn_play_clicked(self):
		if(self.paused):
			self.paused = False
			self.bn_play.setIcon(self.bn_play.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
		else:
			self.paused = True
			self.bn_play.setIcon(self.bn_play.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

	#default functions.


	def reset_widget(self):
		self.camCombo.clear()
		self.camCombo.addItem("Free viewpoint")
		self.camCombo.setCurrentIndex(0)

		self.list_model.clear()
		item = QtWidgets.QTreeWidgetItem()
		item.setText(0,"floor")
		item.setCheckState(0,QtCore.Qt.Checked)
		self.list_model.addTopLevelItem(item)

		self.frame_slider.setMaximum(0)
		self.frame_slider.setValue(0)

		self.frame_id = 0
		self.frame_base	= 0

		self.paused = True

		self.bn_play.setIcon(self.bn_play.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

		self.mainGlWidget.set_render_content([],None)

	def loadDB_core(self):
		#self.reset_widget()
		self.statusbar.showMessage('Data sequence {} loaded'.format(self.seqname))
		#create cameras
		if self.custom_load_cameras is None:
			cam_list = self.default_load_cameras()
		else:
			cam_list = self.custom_load_cameras(self.dataroot,self.seqname)

		self.mainGlWidget.set_cameras(cam_list)
		self.mainGlWidget.element_status.update({'cameras':True})
		

		for hdcam in cam_list:
			self.camCombo.addItem(hdcam['name'])
		self.camCombo.setCurrentIndex(0)


		item = QtWidgets.QTreeWidgetItem()
		item.setText(0,'cameras')
		item.setCheckState(0,QtCore.Qt.Checked)
		self.list_model.addTopLevelItem(item)


		#count frame number
		if self.custom_load_timeline is None:
			frame_num = self.default_load_timeline()
		else:
			frame_num = self.custom_load_timeline(self.dataroot,self.seqname)
		
		self.frame_num = frame_num
		self.frame_slider.setMaximum(self.frame_num)
		self.edit_max_frame.setText('{:d}'.format(self.frame_num))
		self.edit_cur_frame.setText('0')
		self.frame_id = 0

	def loadDB(self):

		#from choose path
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.DirectoryOnly)
		dirname = None
		if dlg.exec_():
			dirname = dlg.selectedFiles()
		#get seqname
		if not dirname:
			return
		self.reset_widget()
		self.dataroot = os.path.dirname(dirname[0])
		self.seqname = os.path.basename(dirname[0])
		#need to firstly clear
		self.loadDB_core()

	

	def insert_person_widgets(self,personID,has_body=True,has_face=False,has_hand=False,has_mesh=False):
		item = QtWidgets.QTreeWidgetItem()
		item.setText(0,'Person:{}'.format(personID))
		item.setCheckState(0,QtCore.Qt.Checked)

		subitems = []
		subitem = QtWidgets.QTreeWidgetItem()
		subitem.setText(0,'body')
		if has_body:
			subitem.setCheckState(0,QtCore.Qt.Checked)
		else:
			subitem.setCheckState(0,QtCore.Qt.Unchecked)
			subitem.setFlags(subitem.flags() ^ QtCore.Qt.ItemIsEnabled)
		subitems.append(subitem)
		subitem = QtWidgets.QTreeWidgetItem()
		subitem.setText(0,'face')

		if has_face:
			subitem.setCheckState(0,QtCore.Qt.Checked)
		else:
			subitem.setCheckState(0,QtCore.Qt.Unchecked)
			subitem.setFlags(subitem.flags() ^ QtCore.Qt.ItemIsEnabled)
		subitems.append(subitem)
		subitem = QtWidgets.QTreeWidgetItem()
		subitem.setText(0,'hand')

		if has_hand:
			subitem.setCheckState(0,QtCore.Qt.Checked)
		else:
			subitem.setCheckState(0,QtCore.Qt.Unchecked)
			subitem.setFlags(subitem.flags() ^ QtCore.Qt.ItemIsEnabled)
		subitems.append(subitem)
		subitem = QtWidgets.QTreeWidgetItem()
		subitem.setText(0,'mesh')
		subitem.setCheckState(0,QtCore.Qt.Checked)
		subitems.append(subitem)
		item.addChildren(subitems)
		self.list_model.addTopLevelItem(item)
	#instead of set frame id, set render options.

	def set_frameid(self,frameid_):
		if(frameid_>self.frame_num-1):
			frameid_ = self.frame_num-1

		if frameid_!=self.frame_id:
			self.frame_id = frameid_
			self.frameIdChanged.emit(frameid_)
			
			if self.custom_load_joints is None:
				joints_total = self.default_load_joints()
			else:
				joints_total = self.custom_load_joints(self.dataroot,self.seqname,self.frame_id)

			if self.custom_load_mesh is None:
				mesh_total = self.default_load_mesh()
			else:
				mesh_total = self.custom_load_mesh(self.dataroot,self.seqname,self.frame_id)
			
			self.edit_cur_frame.setText('{}'.format(self.frame_id))
			
			self.mainGlWidget.set_render_content(joints_total,mesh_total)

	def timerEvent(self,event):
		if(event.timerId()==self.timer.timerId()):
			if(not self.paused):
				frameid_= self.frame_id + 1
				self.set_frameid(frameid_)
				

if __name__=="__main__":
	app = QtWidgets.QApplication(sys.argv)
	ui = PanopticStudioPlayer()
	ui.show()
	sys.exit(app.exec_())  
