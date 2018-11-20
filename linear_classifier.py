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
	count = 0
	with open(trainFile,'r') as f:
		for line in f.readlines():
			filename = line.strip().split()[0]
			image = Image.open(filename).convert('L')
			image = image.resize((32,32)) #To convert the image to greyscale and resize it and flatten
			image = (np.array(image,dtype=np.uint8)).flatten()
			labels = line.strip().split(" ")[1] 
			# print(labels)
			if labels not in mapping:
				mapping[labels] = count
				inverseMapping[count] = labels
				count += 1

			# print(mapping,inverseMapping)
			train_labels.append(mapping[labels])
			imageArray.append(image)
  
	f.close()
	with open(testFile,'r') as f:
		for line in f.readlines():
			filename = line.strip().split()[0]
			image = Image.open(filename).convert('L')
			image = image.resize((32,32)) #To convert the image to greyscale and resize it and flatten
			image = (np.array(image,dtype=np.uint8)).flatten()
			testImageArray.append(image)
			base=os.path.basename(filename).split("_")
			test_labels.append(base[0][2])
	f.close()		
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


def calculateP(X_row, label, w):
	allProducts = np.matmul(w,X_row)
	# print(allProducts)
	max_allProducts = np.max(allProducts)
	for index,word in enumerate(allProducts):
		allProducts[index] = allProducts[index] - max_allProducts
	# print(label)
	correctVal = np.exp(allProducts[label])
	sumVal = 0
	for i in range(total_lables):
		sumVal = np.add(sumVal,np.exp(allProducts[i]))
	return correctVal/sumVal

def softmax_regeression(w):

	r,c = np.shape(new_features)
	# print(r)
	eta = 0.0005
	for itr in range(100):
		mid_w = np.zeros((total_lables,33))
		for i in range(r):
			# print(train_labels)	
			p = calculateP(new_features[i], train_labels[i], w)
			matrix = (1-p)*(np.transpose(new_features[i]))
			for j in range(total_lables):
				if(j == int(train_labels[i])):
					mid_w[j] = np.add(mid_w[j],matrix)
		w = w + eta*mid_w
		

	max_index = -1
	max_value = -1000000000000
	final_label_matrix = np.dot(new_test_features, np.transpose(w))
	count = 0
	for i in range(np.shape(final_label_matrix)[0]):
		max_value = -1000000000000
		max_index = -1
		for index,word in enumerate(final_label_matrix[i]):
			if word  > max_value:
				max_value = word
				max_index = index
			
		# if(max_index == test_labels[i]):
		print(inverseMapping[max_index])

	


if __name__ == "__main__":
	train_labels = []
	test_labels = []
	mapping = {}
	inverseMapping = {}
	trainFile = sys.argv[1]
	testFile = sys.argv[2]
	train_features,test_features = calculate_eigen_pairs(trainFile,testFile)
	train_labels = list(map(int, train_labels))
	test_labels = list(map(int, test_labels))
	classes, counts = np.unique(train_labels, return_counts=True)
	w = np.random.random((len(classes),33))

	new_features = np.c_[train_features,0.01 * np.random.randn(np.shape(train_features)[0])]
	new_test_features = np.c_[test_features,0.01 * np.random.randn(np.shape(test_features)[0])]
	total_lables = len(classes)
	softmax_regeression(w)