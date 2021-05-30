## Library 
import pandas as pd
import numpy as np
import scipy as scipy
from scipy import spatial
import json
import argparse
from argparse import Namespace
import glob

import sklearn as sk
from sklearn import model_selection
from sklearn import metrics
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt 
import time
import os
import datetime
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import cm
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow.keras as keras

## Utils function from https://github.com/ChrisWu1997/2D-Motion-Retargeting
from functional.visualization import motion2video, hex2rgb
from functional.motion import preprocess_motion2d, postprocess_motion2d, openpose2motion
from functional.utils import ensure_dir, pad_to_height
from model import get_autoencoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import config
from dataset import get_meanpose


connect_dict=[[0,1],[1,2],[2,3],[3,4],[4,5],[3,6],[6,7],[7,8],[8,9],[3,10],[10,11],[11,12],[12,13],[0,14],[14,15],[15,16],[16,17],[0,18],[18,19],[19,20],[20,21]]
pose_net_joint=[4,7,8,9,11,12,13,14,15,16,18,19,20]
open_pose_joint=[4,2,11,12,13,7,8,9,0,18,19,20,14,15,16]
win_size=12
output_dir='inc_deep_squat'
task_list=['m01','m02','m03','m04','m05','m06','m07','m08','m09','m10']
s_number=['s01','s02','s03','s04','s05','s06','s07','s08','s09','s10']
e_number=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10']
LR_table=pd.read_csv('LR_table.csv',index_col=0)
kin_position_path='Segmented Movements/Segmented Movements/Kinect/Positions/'
kin_angle_path='Segmented Movements/Segmented Movements/Kinect/Angles'
perfor=pd.DataFrame(columns=task_list,index=s_number)

def to_3D_corrd_in_view(kinect_postion,kinect_angle):
    kinect_postion=np.array(kinect_postion)
    kinect_angle=np.array(kinect_angle)
    kinect_postion=kinect_postion.reshape(22,3)
    kinect_angle=kinect_angle.reshape(22,3)
    for j_p, j_c in connect_dict:
        rot_mat=scipy.spatial.transform.Rotation.from_euler('yxz',kinect_angle[j_p,:3]*np.pi/180).as_matrix()
        kinect_postion[j_c,:]=kinect_postion[j_p,:]+np.matmul(rot_mat,kinect_postion[j_c,:])   
    return kinect_postion

def plot_skelton(dat,**kwargs):

    plt.annotate('Waist',(dat[0,0],-dat[0,1]))
    plt.annotate('head',(dat[4,0],-dat[4,1]))
    plt.annotate('left hand',(dat[9,0],-dat[9,1]))
    plt.annotate('Right hand',(dat[13,0],-dat[13,1]))
    plt.annotate('Left leg',(dat[17,0],-dat[17,1]))
    for j1,j2 in connect_dict:
        plt.plot(dat[[j1,j2],0],-dat[[j1,j2],1],**kwargs)
        
def LR_mirror(kinect_postion,percentage,LR_table,motion): 
    for i in range(len(percentage)):
        if LR_table[motion][percentage.tester[i]]:
            kinect_postion[i,:,0]=-kinect_postion[i,:,0]
    return kinect_postion

def poseNet2openPose(kinect_postion):
    chest=np.mean(kinect_postion[:,[1,4],:],axis=1).reshape(-1,1,3)
    base=np.mean(kinect_postion[:,[7,10],:],axis=1).reshape(-1,1,3)
    new_openpose=kinect_postion[:,[0],:]
    new_openpose=np.concatenate((new_openpose,chest),axis=1)
    new_openpose=np.concatenate((new_openpose,kinect_postion[:,[4,5,6],:]),axis=1)
    new_openpose=np.concatenate((new_openpose,kinect_postion[:,[1,2,3],:]),axis=1)
    new_openpose=np.concatenate((new_openpose,base),axis=1)
    new_openpose=np.concatenate((new_openpose,kinect_postion[:,[10,11,12],:]),axis=1)
    new_openpose=np.concatenate((new_openpose,kinect_postion[:,[7,8,9],:]),axis=1)
    return new_openpose

def getPoseNet(motion):
    return(motion[:,pose_net_joint,:])
def getOpenPose(motion):
    return(motion[:,open_pose_joint,:])
def to_2D(motion):
    return(motion[:,:,:2])

def tl_preprocess_encode(motion,encoder,scale=1.2):
    
    motion=to_2D(motion)
    if motion.shape[1] == 22:
        motion=getOpenPose(motion)
    elif motion.shape[1] == 13:
        motion=poseNet2openPose(motion)
    elif motion.shape[1] == 15:
        motion=motion
    else:
        "Not supported skeleton"
    for i in range(len(motion) - 1, 0, -1):
        motion[i - 1][np.where(motion[i - 1] == 0)] = motion[i][np.where(motion[i - 1] == 0)]

    motion = np.stack(motion, axis=2)
    motion = gaussian_filter1d(motion, sigma=2, axis=-1)
    motion = motion * scale
    
    motion=preprocess_motion2d(motion, mean_pose, std_pose)
    motion=motion.to(config.device)
    return encoder(motion).cpu().detach().numpy().squeeze()

def get_sliding_wins(motion,percentage,win_size):
    percentage.reset_index(inplace=True,drop=True)
    sliding_win=[]
    sliding_percent=[]

    for i in percentage.tester.unique():
        print(i)
        start_id=np.where(percentage.tester == i)[0][0]
        end_id=np.where(percentage.tester == i)[0][-1]+1
        for window in percentage.percentage.iloc[start_id:end_id].rolling(window=win_size,min_periods=win_size):
            if len(window)<win_size:
                continue
            else:
                sliding_win.append(motion[window.index,:,:])
                sliding_percent.append(window.mean())
    sliding_percent=np.array(sliding_percent)
    return(sliding_win,sliding_percent)

args= argparse.Namespace(name='skeleton',model_path='model/pretrained_skeleton.pth',v1='inc_deep_squat',o='inc_deep_squat',gpu_ids=0,w1=720,h1=720,transparency=False,save_frame=1,
                         fps=25,color1='#a50b69#b73b87#db9dc3',max_len=480,max_frame=480)
config.initialize(args)
mean_pose, std_pose = get_meanpose(config)
net = get_autoencoder(config)
net.load_state_dict(torch.load(args.model_path))
net.to(config.device)
net.eval()
encoder=nn.Sequential(*list(net.mot_encoder.children())[0][0:6])

for task in task_list:
    
    data=np.load('data/'+task+'.npy')
    percentage=pd.read_csv('data/'+task+'.csv')
    data=LR_mirror(data,percentage,LR_table,'m01')
    for subj in s_number:
        print("Motion: %s"%task)
        print('Testing on subject: %s'%subj)
        X_train, y_train = np.array(data)[np.where(percentage['tester']!=subj),:,:][0],percentage.iloc[np.where(percentage['tester']!=subj)[0],:]
        X_test, y_test = np.array(data)[np.where(percentage['tester']==subj),:,:][0],percentage.iloc[np.where(percentage['tester']==subj)]

        sliding_win, sliding_percent=get_sliding_wins(X_train, y_train,win_size)
        print("Train sliding window length: %s"%len(sliding_win))
        encode=[]
        for i in range(len(sliding_win)):
            data=tl_preprocess_encode(sliding_win[i],encoder=encoder)
            encode.append(data)
        encode=np.array(encode)
        X_train, y_train =sk.utils.shuffle(encode,sliding_percent)
        model = keras.models.Sequential()
        model.add(keras.Input(shape=(96,3,)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv1D(32, 5, activation='relu'))
        model.add(keras.layers.MaxPooling1D(2))
        model.add(keras.layers.Conv1D(64, 3, activation='relu'))
        model.add(keras.layers.MaxPooling1D(2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(8))
        model.add(keras.layers.Dense(4))
        model.add(keras.layers.Dense(1,activation='linear'))
        model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mean_squared_error",
        metrics=['MeanSquaredError']
        )
        print(model.summary())
        model.fit(x=X_train.astype('float32'),y=y_train.astype('float32'), epochs=500,validation_split=0.2,batch_size=50)
        model.save('model/Motion_'+task+"_testing on_"+subj+'.h5')
        
        sliding_win, sliding_percent=get_sliding_wins(X_test, y_test,win_size)
        encode_test=[]
        for i in range(len(sliding_win)):
            data=tl_preprocess_encode(sliding_win[i],encoder=encoder)
            encode_test.append(data)
        encode_test=np.array(encode_test)
        predict=model.predict(encode)
        perfor[task][subj]=sk.metrics.mean_squared_error(sliding_percent,predict)
        fig, ax= plt.subplots()
        ax.set_title("Motion: "+task+" testing on "+subj)
        ax.plot(predict)
        ax.plot(sliding_percent)
        fig.savefig("test/Transfer_motion "+task+" testing on "+subj+'.png')
perfor.to_csv("Performance_report_transfer_learning.csv")