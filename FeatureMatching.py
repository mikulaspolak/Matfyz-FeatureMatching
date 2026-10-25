# source: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
# downgrade OpenCV a bit since some none-free features are not avilable
# pip install opencv-python==3.4.2.17
# pip install opencv-contrib-python==3.4.2.17

import cv2 as cv
import matplotlib.pyplot as plt
import sys

priecinok = 'izba'

img1 = cv.imread(priecinok + '/img1.png', 0)  # queryimage # left image
img2 = cv.imread(priecinok + '/img2.png', 0)  # trainimage # right image


def BruteForceORB():
	print("-> Starting BruteForceORB....")
	# Initiate ORB detector
	orb = cv.ORB_create()
	# find the keypoints and descriptors with ORB
	kp1, des1 = orb.detectAndCompute(img1, None)
	kp2, des2 = orb.detectAndCompute(img2, None)

	# create BFMatcher object
	bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
	# Match descriptors.
	matches = bf.match(des1, des2)
	# Sort them in the order of their distance.
	matches = sorted(matches, key=lambda x: x.distance)

	# Finally, let`s draw matching pairs
	draw_params = dict(matchColor=(0, 255, 0),
	                   singlePointColor=(255, 0, 0),
	                   flags=cv.DrawMatchesFlags_DEFAULT)
	img_out = cv.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
	plt.clf(), plt.imshow(img_out)
	plt.axis('off')
	#plt.title("Brute-Force Matching with ORB Descriptors")
	fig1 = plt.gcf()
	fig1.canvas.start_event_loop(sys.float_info.min)  # for Exception in Tkinter callback
	plt.savefig(priecinok + '/matching_BF_ORB.png', bbox_inches='tight', dpi=200)


def BruteForceSIFT():
	print("-> Starting BruteForceSIFT....")  # Initiate SIFT detector
	sift = cv.xfeatures2d.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)
	# BFMatcher with default params
	bf = cv.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)
	# Apply ratio test
	good = []
	for m, n in matches:
		if m.distance < 0.7 * n.distance:
			good.append([m])

	# Finally, let`s draw matching pairs
	draw_params = dict(matchColor=(0, 255, 0),
	                   singlePointColor=(255, 0, 0),
	                   flags=cv.DrawMatchesFlags_DEFAULT)
	img_out = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, **draw_params)
	plt.clf(), plt.imshow(img_out)
	plt.axis('off')
	#plt.title("Brute-Force Matching with SIFT Descriptors and Ratio Test")
	fig1 = plt.gcf()
	fig1.canvas.start_event_loop(sys.float_info.min)  # for Exception in Tkinter callback
	plt.savefig(priecinok + '/matching_BF_SIFT.png', bbox_inches='tight', dpi=200)


def Flann():
	print("-> Starting Flann....")
	# Initiate SIFT detector
	sift = cv.xfeatures2d.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1, None)
	kp2, des2 = sift.detectAndCompute(img2, None)
	# FLANN parameters
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)  # or pass empty dictionary
	flann = cv.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)
	# Need to draw only good matches, so create a mask
	matchesMask = [[0, 0] for i in range(len(matches))]
	# ratio test as per Lowe's paper
	for i, (m, n) in enumerate(matches):
		if m.distance < 0.7 * n.distance:
			matchesMask[i] = [1, 0]

	# Finally, let`s draw matching pairs
	draw_params = dict(matchColor=(0, 255, 0),
	                   singlePointColor=(255, 0, 0),
	                   matchesMask=matchesMask,
	                   flags=cv.DrawMatchesFlags_DEFAULT)
	img_out = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
	plt.clf(), plt.imshow(img_out)
	plt.axis('off')
	#plt.title("FLANN based Matcher with SIFT Descriptors and Ratio Test")
	fig1 = plt.gcf()
	fig1.canvas.start_event_loop(sys.float_info.min)  # for Exception in Tkinter callback
	plt.savefig(priecinok + '/matching_FLANN.png', bbox_inches='tight', dpi=200)


BruteForceORB()
BruteForceSIFT()
Flann()
