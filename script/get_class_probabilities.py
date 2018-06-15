#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - February 2018
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
from argparse import ArgumentParser
import numpy as np
import scipy.signal
import soundfile
import librosa.feature


def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-w', dest='wave_dir', required=True)
    a.add_argument('-l', dest='lab_dir', required=True)
    a.add_argument('-o', dest='outdir', required=True)        
    a.add_argument('-s', dest='smooth_length', required=False, default=600, type=int, help='Length of smoothing window (in frames) for class posteriors')
    opts = a.parse_args()
    
    # ===============================================

    wave_dir = opts.wave_dir
    lab_dir = opts.lab_dir
    outdir = opts.outdir
    smooth_length = opts.smooth_length

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # categories = ['s','ms','m','p']
    # categories_full = ['speech','speech & music','music','report']

    features, labels, unlabelled, unlabelled_names = load_data(wave_dir, lab_dir)

    print 'build forest of trees....'
    from sklearn.ensemble import RandomForestClassifier
    
    clf = RandomForestClassifier(max_depth=10, random_state=0)
    clf.fit(features, labels)
    print clf
    classes = ' '.join( clf.classes_ )
    print 'apply to new data'

    nepisodes = len(unlabelled) # to debug, set to e.g.: 2

    for episode in range(nepisodes):
        
        guess = clf.predict_proba(unlabelled[episode])
        smooth_guess = smooth_posteriors(guess, smooth_length)

        outstem = os.path.join(outdir, unlabelled_names[episode])
        np.savetxt(outstem + '.csv', guess, header=classes, fmt='%.4e')
        np.savetxt(outstem + '_smoothed.csv', smooth_guess,  header=classes, fmt='%.4e')

        print 'Wrote ' + outstem + '_smoothed.csv'

def smooth_posteriors(posteriors, smooth_length):
    filt = np.hanning(smooth_length) / np.hanning(smooth_length).sum()
    posteriors_smooth = np.zeros(posteriors.shape)
    m,n = posteriors.shape
    for i in range(n):
        smoothed = scipy.signal.convolve(posteriors[:,i], filt, mode='same')
        posteriors_smooth[:,i] = smoothed
    return posteriors_smooth
    

def read_labels(fname):
    
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    lines = [line.strip(' \n').split(' ') for line in lines]
    total_frames = 0
    frames_cats = []
    for (start, end, category) in lines:
        
        seconds = float(end) - float(start)
        frames = int(seconds * 100.0)
        frames_cats.append((frames, category))

        total_frames += frames

    data = []
    for (frames, cat) in frames_cats:
        data.extend([cat] * frames)
        
    return np.array(data)



def load_data(wave_dir, lab_dir):

    assert len(glob.glob(lab_dir + '/*.lab')) > 0, 'no labels in %s'%(lab_dir)

    waves = glob.glob(wave_dir + '/*.wav') 

    feats_only = []
    unlabelled_names = []
    labelled_feats = []
    all_labels = []

    for wave in waves:
        _, base = os.path.split(wave)
        base = base.replace('.wav','')
        print 'loading %s'%(base)

        lab = os.path.join(lab_dir, base + '.lab')

        waveform, sr = soundfile.read(wave)
        features = librosa.feature.mfcc(y=waveform, sr=sr, dct_type=3, n_mfcc=13, hop_length=int(sr*0.01)) ## HTK style mfccs, 10 ms frameshift

        # print features.shape
        features = features.transpose()

        feats_only.append(features)
        unlabelled_names.append(base)

        if not os.path.isfile(lab):
            continue
        
        #print 'getting label for %s'%(base)

        labels = read_labels(lab)

        feat_frames,n = features.shape
        lab_frames, = labels.shape

        
        frames = min(feat_frames, lab_frames)
        
        features = features[:frames,:]
        labels = labels[:frames]

        labelled_feats.append(features)
        all_labels.append(labels)

    labelled_feats = np.vstack(labelled_feats)
    all_labels = np.concatenate(all_labels)

    return labelled_feats, all_labels, feats_only, unlabelled_names

if __name__=="__main__":
    main_work()

