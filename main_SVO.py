import cv2
import numpy as np
import glob
import math
import SVO_icp
import matplotlib.pyplot as plt
import time


def features(lp,lc):
	fast = cv2.FastFeatureDetector_create(threshold=50)
	orb = cv2.ORB_create()		# Feature detection using ORB

	kp_left_prev = fast.detect(lp, None)
	kp_left_cur = fast.detect(lc, None)

	kp_left_prev,des_left_prev = orb.compute(lp,kp_left_prev)
	kp_left_cur,des_left_cur = orb.compute(lc,kp_left_cur)

	return kp_left_prev,kp_left_cur,des_left_prev,des_left_cur


def reduced_matches(kp_left_prev,kp_left_cur,des_left_prev,des_left_cur):

	bf = cv2.BFMatcher(cv2.NORM_HAMMING)	# BF matcher
	matches = bf.match(des_left_prev, des_left_cur)
	good_matches = []
	min_dist =10000
	max_dist = 0


	for m in matches:
			if m.distance <min_dist:
				min_dist = m.distance
			if m.distance > max_dist:
				max_dist = m.distance
	for m in matches:
		if m.distance <= max(2*min_dist,30.0):
			good_matches.append(m)

	
	plp = []
	plc = []
	klp = []
	klc = []
	dlp = []
	dlc = []
	for m in good_matches:
		klp.append(kp_left_prev[m.queryIdx])
		klc.append(kp_left_cur[m.trainIdx])
		dlp.append(des_left_prev[m.queryIdx])
		dlc.append(des_left_cur[m.trainIdx])
		plp.append(kp_left_prev[m.queryIdx].pt)
		plc.append(kp_left_cur[m.trainIdx].pt)


	plp = np.asarray(plp)
	plc = np.asarray(plc)
	klp = np.array(klp)
	klc = np.array(klc)
	dlp = np.array(dlp)
	dlc = np.array(dlc)

	return klp,klc,dlp,dlc,plp,plc,good_matches


def poseEstimation(plp,plc,inlier=False):

	E, mask = cv2.findEssentialMat(plp,plc,focal=focal,pp=pp,method=cv2.RANSAC,prob=0.999,threshold=1.0)
	if inlier==True:
		plp = plp[mask.ravel() == 1]
		plc = plc[mask.ravel() == 1]

	_, R, t, _ = cv2.recoverPose(E, plp, plc,focal=focal,pp=pp)

	return R,t,plp,plc,mask

def reduced_features(klp,klc,dlp,dlc,plp,plc):

	_, mask = cv2.findEssentialMat(plp,plc,focal=focal,pp=pp,method=cv2.RANSAC,prob=0.999,threshold=1.0)

	plp = plp[mask.ravel() == 1]
	plc = plc[mask.ravel() == 1]

	klp = klp[mask.ravel() == 1]
	klc = klc[mask.ravel() == 1]

	dlp = dlp[mask.ravel() == 1]
	dlc = dlc[mask.ravel() == 1]

	return klp,klc,dlp,dlc,plp,plc


def triangulation(pts1,pts2,lpm,rpm):

	points = cv2.triangulatePoints( lpm[:3], rpm[:3], pts1.T, pts2.T )
	#points = points/points[3]

	X = points[0,:]
	Y = points[1,:]
	Z = points[2,:]
	D = points[3,:]
	return X,Y,Z,D

def right_prev(right_image_prev):

	fast = cv2.FastFeatureDetector_create(threshold=50)#, nonmaxSuppression=True)
	orb = cv2.ORB_create()		# Feature detection using ORB

	kp_right_prev = fast.detect(right_image_prev, None)
	kp_right_prev,des_right_prev = orb.compute(right_image_prev,kp_right_prev)

	return kp_right_prev,des_right_prev

def reduced_matches2(plpi,kp_left_prev,kp_right_prev,dlpi,des_right_prev,lpm,rpm,X1,Y1,Z1,D1):

	bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)	# BF matcher method for matching descriptors
	matches = bf.match(dlpi,des_right_prev)

	masker = np.zeros(len(kp_left_prev))

	min_dist =10000
	max_dist = 0

	klp = []
	krp = []
	plp = []
	prp = []

	for m in matches:
			if m.distance <min_dist:
				min_dist = m.distance
			if m.distance > max_dist:
				max_dist = m.distance
	count = 0
	good_matches = []

	for m in matches:
		if m.distance <= max(2*min_dist,30.0):
			masker[count]=1
			good_matches.append(m)
		count = count+1

	print("lenght of matches from left to right prev:",len(good_matches))

	X1 = X1[masker.ravel() == 1]
	Y1 = Y1[masker.ravel() == 1]
	Z1 = Z1[masker.ravel() == 1]
	D1 = D1[masker.ravel() == 1]
	
	for m in good_matches:

		klp.append(kp_left_prev[m.queryIdx])
		krp.append(kp_right_prev[m.trainIdx])
		plp.append(kp_left_prev[m.queryIdx].pt)
		prp.append(kp_right_prev[m.trainIdx].pt)

	klp = np.array(klp)
	krp = np.array(krp)
	plp = np.array(plp)
	prp = np.array(prp)

	#print("lenght of the plp before:",len(plp))
	_,_,plp,prp,mask = poseEstimation(plp,prp,inlier=True)

	#print("phaseII:lenght of first 3D point updated before:",len(X1))
	X1 = X1[mask.ravel() == 1]
	Y1 = Y1[mask.ravel() == 1]
	Z1 = Z1[mask.ravel() == 1]
	D1 = D1[mask.ravel() == 1]
	#print("phaseII:lenght of first 3D point updated after:",len(X1))

	#print("lenght of the plp after:",len(plp))
	X2,Y2,Z2,D2 = triangulation(plp,prp,lpm,rpm)
	#print("lenght of right points:",np.shape(X2))

	return X1,Y1,Z1,X2,Y2,Z2,matches,D1,D2

def getscale(X1,Y1,Z1,X2,Y2,Z2):
	
	scales= []

	for i in range(len(X1)):  # Taking length of 3D set points
		
		x1 = X1[i]
		y1 = Y1[i]
		z1 = Z1[i]

		x2 = X2[i]
		y2 = Y2[i]
		z2 = Z2[i]

		Pmono = math.sqrt(x1**2 + y1**2 + z1**2)
		Pster = math.sqrt(x2**2 + y2**2 + z2**2)

		scale = Pmono/Pster

		if not (math.isnan(scale)): # checking for NaN's
			scales.append(scale)

	actual_scale = np.mean(scales)  # Mean of scales, mean seems to have better results than Median
	
	return actual_scale


def groundTruth(frame_id):      # Prints the ground translations on the GUI image for reference 
	g_pose = open('/home/bhuvan/Downloads/IndependentStudy/dataset/dataset/sequences/00/poses.txt','r').readlines()
	ss = g_pose[frame_id].strip().split()
	x = float(ss[3])
	z = float(ss[11])
	return x,z

def absolutescale(i):

	g_pose = open('/home/bhuvan/Downloads/IndependentStudy/dataset/dataset/sequences/00/poses.txt','r').readlines()
	ss = g_pose[i-1].strip().split()
	x_prev = float(ss[3])
	y_prev = float(ss[7])
	z_prev = float(ss[11])
	ss = g_pose[i].strip().split()
	x = float(ss[3])
	y = float(ss[7])
	z = float(ss[11])

	return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))

def main():

	# Reading raw data Kitti dataset images 	
	left_imgList = [] 			   # Create an empty list to stack image paths
	right_imgList = []
	strLine  = '/home/bhuvan/Downloads/IndependentStudy/dataset/dataset/sequences/00/image_'
	left_imgpath = strLine + str(0) + '/*'
	right_imgpath = strLine + str(1) + '/*'

	# strLine  = '/home/bhuvan/Desktop/stereo/submissions/videos/'
	# left_imgpath = strLine + 'left' + '/*'
	# right_imgpath = strLine + 'right' + '/*'

	for i in glob.glob(left_imgpath):
		left_imgList.append(i)

	for i in glob.glob(right_imgpath):
		right_imgList.append(i)

	left_imgList=sorted(left_imgList)
	right_imgList=sorted(right_imgList)

	# variable declaration

	lpm = np.array([[718.856,0,607.1928,0],[0,718.56,185.2157,0],[0,0,1,0]])         # Kiti Left projection matrix declaration 
	rpm = np.array([[718.856,0,607.1928,-386.1448],[0,718.56,185.2157,0],[0,0,1,0]]) # Kitti Right projection matrix declaration


	# lpm = np.array([[629.37048544,0,337.68033869,0],[0,625.60905585,212.92168682,0],[0,0,1,0]])         # Kiti Left projection matrix declaration 
	# rpm = np.array([[629.37048544,0,337.68033869,142.24],[0,625.60905585,212.92168682,0],[0,0,1,0]])  # Kitti Right projection matrix declaration
	
	gui_img = np.zeros((920,920,3)) 				#Create a empty image 
	
	T_cur = np.eye(4)  								# Create a an empty cummulative translation and rotation 
	cum_R = np.eye(3)
	cum_t = np.reshape(np.zeros((3,1)), (3,1))
	cum_tp = [[0],[0],[0]]
	

	U=0 						# miscellaneous data declaration
	t_prev = np.zeros((3,1))
	s_cum = 0
	frame_rate = 30
	prev_time = 0

	for i,(l,r) in enumerate(zip(left_imgList,right_imgList)):

		if i==0:
			left_image_prev = cv2.imread(l)    # For the first iteration it stores as previous frames.
			right_image_prev = cv2.imread(r)
			continue

		left_image_cur = cv2.imread(l) 		   # From the second iteration 
		right_image_cur = cv2.imread(r)
		
		time_elapsed = time.time() - prev_time  
		if time_elapsed> 1./frame_rate:
			prev_time = time.time()

			cv2.imshow("left",left_image_cur)  # printing the raw images
			cv2.imshow("right",right_image_cur)
			#cv2.waitKey(0)
			kp_left_prev,kp_left_cur,des_left_prev,des_left_cur = features(left_image_prev,left_image_cur)   # Feature Detection step
			
			klp,klc,dlp,dlc,plp,plc,good_matches = reduced_matches(kp_left_prev,kp_left_cur,des_left_prev,des_left_cur) # Filteration Stage - I
			
			klpi,klci,dlpi,dlci,plpi,plci = reduced_features(klp,klc,dlp,dlc,plp,plc)  									# Inlier filteration Stage - II
	
			R,t,plp,plc,_ = poseEstimation(plp,plc)																	# Pose Estimation Stage
		
			lpm_nextFrame = lpm.copy()
			lpm_nextFrame[:3,:3] = lpm_nextFrame[:3,:3] @ R
			lpm_nextFrame[0:3,:]=t
			
			X1,Y1,Z1,D1 = triangulation(plpi,plci,lpm,lpm_nextFrame)  # These 3D points will have unknown scale factor
			print("Inliers produced from left side:",np.shape(klpi),np.shape(klci),np.shape(dlpi),np.shape(dlci),np.shape(plpi),np.shape(plci),np.shape(X1),np.shape(Y1),np.shape(Z1))
			print("All of the above variables will have same length")

			kp_right_prev,des_right_prev = right_prev(right_image_prev) # Right image feature extraction

			X1,Y1,Z1,X2,Y2,Z2,matches,D1,D2 = reduced_matches2(plpi,klpi,kp_right_prev,dlpi,des_right_prev,lpm,rpm,X1,Y1,Z1,D1)	# Filteration Stage - III

			print("ALL 3D POINT SHAPES:",np.shape(X1),np.shape(Y1),np.shape(Z1),np.shape(X2),np.shape(Y2),np.shape(Z2))   
			print("All the six 3D points have the same length")

			# Image Visualization stage, this would be uncommented during 3D visualization

			# fig = plt.figure()
			# ax = fig.add_subplot(111, projection='3d')
			# ax.scatter(list(X1), list(Y1), list(Z1),color='red')
			# ax.scatter(list(X2), list(Y2), list(Z2),color='blue')
			# ax.set_xlabel('X Label')
			# ax.set_ylabel('Y Label')
			# ax.set_zlabel('Z Label') 
			# plt.show()
			
			scaler = getscale(X1,Y1,Z1,X2,Y2,Z2) # Calculating the scale using stereo camera.
			#scaler = absolutescale(i)  		  # Calculating the scale using monocular camera.
			if i-1==0:
				cum_R=R
				cum_t=t
			elif scaler is not None:
				scaler = scaler
				cum_t = cum_t + scaler*(cum_R.dot(t))
				cum_R = R.dot(cum_R)

			
			del_t = np.sqrt((cum_t[0] - cum_tp[0])**2 + (cum_t[1] - cum_tp[1])**2 + (cum_t[2] - cum_tp[2])**2)

			if del_t > 0.0:  # if the camera moves over certain distance
				true_x,true_z = groundTruth(i)
				cv2.circle(gui_img, (int(cum_t[0]) + 500, int(cum_t[2]) + 600), 2, (255, 0, 0), 2)
				#cv2.circle(gui_img, (int(true_x) + 500, -int(true_z) + 600), 2, (0, 255, 0), 2)
				
				img_text = gui_img.copy()
				text = "Coordinates: x=%1fm y=%1fm z=%1fm"%(cum_t[0],cum_t[1],cum_t[2])
				img_text = cv2.putText(img_text, text, (20,80), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
				
				text1 = "Number of left-to-left matches:%1i"%(len(good_matches))
				img_text = cv2.putText(img_text, text1, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

				text2 = "Number of left-to-right matches:%1i"%(len(matches))
				img_text = cv2.putText(img_text, text2, (20,60), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

				text3 = "Scale:%1f"%(scaler)
				img_text = cv2.putText(img_text, text3, (20,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

				cv2.imshow("gui",img_text)
				cum_tp = cum_t
				left_image_prev = left_image_cur   # Next iteration current left and right image will previous left & right 
				right_image_prev = right_image_cur
				#cv2.waitKey(0)
		print("------------------------------------------------------")
		key_pressed = cv2.waitKey(1) & 0xFF
		if key_pressed == ord('q') or key_pressed == 27:
				break


if __name__ == '__main__':


	global focal,pp,baseline

	## Generic dataset - kitti dataset parametes 
	focal = 718.8560
	pp = (607.1928, 185.2157)
	baseline = -1*(-386.1448/focal) #in cm

	# # Custom dataset parameters - microsoft webcam
	# focal = 629.37048544
	# pp = ( 337.68033869,212.92168682)
	# baseline = 14.224 #in cm

	main() 

