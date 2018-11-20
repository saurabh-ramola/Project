import numpy as np
import os
import sys
import glob
from PIL import Image
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
def calculate_eigen_pairs(trainFile,testFile):
	path = './dataset'
	count = 0
	imageArray = []
	testImageArray = []
	with open(trainFile,'r') as f:
		for line in f.readlines():
			filename = line.strip().split()[0]
			image = Image.open(filename).convert('L')
			image = image.resize((32,32)) #To convert the image to greyscale and resize it
			image = np.array(image,dtype=np.uint8)
			image = image.flatten()
			imageArray.append(image)
			labels = line.strip().split(" ")[1] 
			# print(labels)
			if labels not in mapping:
				mapping[labels] = count
				inverseMapping[count] = labels
				count += 1
			# print(mapping,inverseMapping)
			# print(mapping,inverseMapping)
			train_labels.append(mapping[labels])
  
	f.close()
	with open(testFile,'r') as f:
		for line in f.readlines():
			filename = line.strip().split()[0]
			image = Image.open(filename).convert('L')
			image = image.resize((32,32)) #To convert the image to greyscale and resize it
			image = np.array(image,dtype=np.uint8)
			image = image.flatten()
			testImageArray.append(image)
			base=os.path.basename(filename).split("_")
			test_labels.append(base[0][2])
			
	imageArray = np.array(imageArray)
	imageMean = imageArray.mean(axis = 0)
	imageArray = imageArray - imageMean
	cov_mat = np.cov(np.transpose(imageArray))
	eig_vals, eig_vecs = np.linalg.eig(cov_mat)
	
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
	eig_pairs.sort(key=lambda x: x[0],reverse = True)
	X = eig_pairs[0:32]
	x_array = [x[1] for x in X]
	matrix_w = np.hstack([m.reshape(1024,1) for m in x_array])
	Y = np.dot(np.transpose(matrix_w),np.transpose(imageArray))
	Z = np.dot(np.transpose(matrix_w),np.transpose(testImageArray - imageMean))
	return np.transpose(Y),np.transpose(Z)




def gaussian(x, m, v):
	return np.log((1.0/(np.sqrt(1.0/2*np.pi*v*v)))*np.exp( -1.0*(((x - m)/v)**2) ))

def get_likelihood(point, means, stddev):
	feat_prob = np.zeros((num_feats, num_classes))
	for y in classes:
		for i in range(num_feats):
			feat_prob[i, int(y)] = gaussian(point[i], means[i, int(y)], stddev[i, int(y)]) # get the probability
	likelihood = np.zeros((num_classes, 1)) # likelihood for each class 'y'
	for y in classes:
		for i in range(32):
			if(feat_prob[i,int(y)]):
				likelihood[int(y)] += feat_prob[i, int(y)] # mutliply for each feature 'x_{i}'
	return likelihood

def predict(point, means, stddev, prior,testClasses):
	likelihood = get_likelihood(point, means, stddev)
	posterior = []
	for i in range(num_classes):
		posterior.append(prior[i] + likelihood[i])
	prediction = np.argmax(posterior)
	return prediction

def print_predictions():
	predictions = []
	testClasses, testCounts = np.unique(test_labels, return_counts=True)
	count = 0

	for i in range(len(test_labels)):
		prediction = predict(test_features[int(i), :], means, stddev, prior,testClasses)
		predictions.append(prediction)
		print(inverseMapping[prediction])
	
	
		
if __name__ == "__main__":
	train_labels = []
	test_labels = []
	trainFile = sys.argv[1]
	testFile = sys.argv[2]
	mapping = {}
	inverseMapping = {}
	# print(trainFile,testFile)
	train_features,test_features = calculate_eigen_pairs(trainFile,testFile)

	"""Calculating Prior Probabilities"""
	classes, counts = np.unique(train_labels, return_counts=True)

	num_classes = len(classes)
	num_feats = train_features[0, :].shape[0]

	prior = np.array([ x*1.0/len(train_labels) for x in counts ])
	means = np.zeros((num_feats, num_classes)) # every feature, for each class
	stddev = np.zeros((num_feats, num_classes)) # every feature, for each class
	
	for y in classes: # selecting y
		find_labels = []
		for index,clas in enumerate(train_labels):
			if clas == y:
				find_labels.append(index)
		pts = train_features[find_labels, :]
		for i in range(pts.shape[1]):
			means[i, int(y)] = np.mean(pts[:, i])
			stddev[i, int(y)] = np.std(pts[:, i])
	print_predictions()