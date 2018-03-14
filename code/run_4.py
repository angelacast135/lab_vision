
 ################## This is run.py for Segmentation assignenmt of IBIO4680 ################## 

import os
import time
import numpy as np
import random
import pickle
import glob
import pdb

# matplotlib inline
import matplotlib.pyplot as plt
# OpenCV packages
# normal installation routine


import cv2
# for reading .mat files
import scipy.io as spio
from skimage import io, color

ims_dic = {'name': '', 'image':[]}
def read_dataset():
    
    for i, file in enumerate(glob.glob(os.path.join("../BSR/BSDS500/data/images/train/*.jpg"))):
        
        image = cv2.imread(file,1) 
        file_name = file[33:-4]
        ims_dic[i] = {'name': file_name, 'image':image}
        length_fold=i+1

    return ims_dic, length_fold

def k_means( im , k  ):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn import preprocessing

    im_size = im.shape

    #reshape image conserving colour channels and given the case x,y features
    im_vector=np.reshape(im, (-1,im_size[-1]))

    #now normalize image for posing the problem, this scale transform x\in R^n  with \mu_x=0, std_x=1 
    im_vector = preprocessing.scale(im_vector)
            
    #print(k)
    #print(type(k))
    #print(np.int8(k))
    #print(type(np.int8(k)))

    k_im = KMeans(n_clusters=np.uint8(k), init='random').fit( im_vector ) 
    im_segmented = k_im.predict(im_vector)
    im_segmented = np.reshape(im_segmented , (im_size[0],im_size[1]) )

    return im_segmented


def watershed_segmentation(im, n_clusters):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi
    from sklearn import preprocessing
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max

    im = np.mean(im,axis=2)
    local_maxi = peak_local_max(im, indices=False, num_peaks=n_clusters, num_peaks_per_label=1)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-im, markers, mask=im)

    return labels

def add_position_feature(image):
    im_size = np.shape(image)
    x_idx = np.array(range(im_size[0]))
    x_idx = np.matlib.repmat(x_idx, im_size[1],1)
    x_idx = np.transpose(x_idx)

    y_idx = np.array(range(im_size[1]))
    y_idx = np.matlib.repmat(y_idx, im_size[0] , 1)

    im_xy = np.dstack((image,x_idx,y_idx))

    return im_xy


#numberOfClusters positvie integer (larger than 2)
def segmentByClustering( rgbImage, featureSpace, clusteringMethod, numberOfClusters):
    

    #featureSpace : 'rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy' or 'hsv+xy'
    if featureSpace == 'rgb':
        im_feat = rgbImage
    
    elif featureSpace == 'lab': 
        im_feat = color.rgb2lab(rgbImage)

    elif featureSpace == 'hsv':
        im_feat = color.rgb2hsv(rgbImage)

    elif featureSpace == 'rgb+xy':
        im = rgbImage
        im_feat = add_position_feature(im)

    elif featureSpace == 'lab+xy':
        im = color.rgb2lab(rgbImage)
        im_feat = add_position_feature(im)


    elif featureSpace == 'hsv+xy':
        im = color.rgb2hsv(rgbImage)
        im_feat = add_position_feature(im)

    else:
        # default assume feature space is LAB 
        featureSpace = 'lab'
        im_feat = color.rgb2lab(rgbImage)


    #clusteringMethod = 'kmeans', 'gmm', 'hierarchical' or 'watershed'.

    if clusteringMethod == 'kmeans':
        #return vector of segmentaion 
        im_segmented = k_means(im_feat,numberOfClusters)
    
    elif clusteringMethod == 'watershed':
        im_segmented = watershed_segmentation(im_feat,numberOfClusters)
    
    else:
        #by default assume kmeans is used
        clusteringMethod == 'kmeans'
        im_seg = k_means(im_feat,numberOfClusters)


    return im_segmented

if  os.path.isdir("../BSR"):
    print('Reading BSDS ...')
    ims_dic, length_fold = read_dataset()
else:
    print('BSDS500 dataset is downlading...')
    os.system('wget http://157.253.63.7/BSDS500FastBench.tar.gz')
    os.system('tar -xvzf ./BSDS500FastBench.tar.gz')
    os.system('mv  ./BSR ../BSR')
    os.system('rm ./BSR')
    ims_dic, length_fold = read_dataset()

if not  os.path.isdir("./train"):
    os.system('mkdir ./train')
    os.system('mkdir ./train/kmeans')
    os.system('mkdir ./train/kmeans/rgb+xy')
    os.system('mkdir ./train/kmeans/lab+xy')
    os.system('mkdir ./train/kmeans/hsv+xy')
    os.system('mkdir ./train/watershed')
    os.system('mkdir ./train/watershed/rgb+xy')
    os.system('mkdir ./train/watershed/lab+xy')
    os.system('mkdir ./train/watershed/hsv+xy')


featureSpaces =   [ 'rgb+xy', 'lab+xy'] #, 'hsv+xy']
clusteringMethods = ['kmeans', 'watershed']
numberOfClusters = [2,4,6,8,10,12,14,16,18,20]

#pdb.set_trace()
# TRAIN
for i in range(length_fold): #range(4): 
    im = ims_dic[i]['image']
    file_name = ims_dic[i]['name']
    print('-------------we are in image %d---------------'% i)

    for ff , featureSpace in enumerate(featureSpaces):
        for cc, clusteringMethod in enumerate(clusteringMethods):
            obj_arr = np.zeros((len(numberOfClusters),), dtype=np.object)
            print('File to save: '+'./'+clusteringMethod+'/'+featureSpace+'/'+file_name)
            for kk, n_cluster in enumerate(numberOfClusters):

                print("The number of cluster in current image is {}".format(n_cluster))
                #print('we are in image %d'% i)
                
                im_seg = segmentByClustering( im, featureSpace , clusteringMethod , n_cluster)
                im_seg_u16 = np.array(im_seg, dtype=np.uint16)
                obj_arr[kk] = im_seg_u16

            spio.savemat('./train/'+clusteringMethod+'/'+featureSpace+'/'+file_name+'.mat', {'segs':obj_arr})
