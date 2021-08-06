import numpy as np
import scipy as scipy
from scipy import signal
import argparse
from argparse import Namespace
import sklearn as sk
from sklearn import metrics

def repcount(dat,j_d,fs,wins,noverlap):
    dat=np.load(dat,allow_pickle=True)
    sim=sk.metrics.pairwise.cosine_similarity(dat.reshape(-1,j_d))
    f, t, Sxx = scipy.signal.spectrogram(np.pad(sim[0],wins,mode='edge'),window=('boxcar'),fs=fs,nperseg=wins,noverlap=wins-noverlap)
    c_hat=np.ceil(sum(f[np.argmax(Sxx,axis=0)[np.where(t>wins/fs)[0][0]:np.where(t<(len(sim[0])/fs+wins/fs))[0][-1]]]*(t[1]-t[0])))
    print("Estimated count %.0f"%c_hat)
    return(c_hat)


if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--data",type=str, help="path to skeleton data")
    a.add_argument("-j",default=18,type=int, help="Joint number of the skeleton")
    a.add_argument("-d",default=2,type=int, help="Dimension of the joint")
    a.add_argument("-f",default=30,type=int, help="Video frequency")
    a.add_argument("--wins",default=256,type=int, help="Sliding window frequency")
    a.add_argument("--noverlap",default=1,type=int, help="Sliding window steps")

    args = a.parse_args()
    print(args)
    repcount(args.data, args.j*args.d, args.f,args.wins,args.noverlap)
    
    