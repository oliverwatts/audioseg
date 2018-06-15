#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: Natural Speech Technology - February 2015 - www.natural-speech-technology.org
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
import fileinput
from argparse import ArgumentParser

import math
# from lxml import etree

## Check required executables are available:
# 
# from distutils.spawn import find_executable
# 
# required_executables = ['sox', 'ch_wave']
# 
# for executable in required_executables:
#     if not find_executable(executable):
#         sys.exit('%s command line tool must be on system path '%(executable))
    
import numpy as np
import scipy.signal

from speech_manip import get_speech
import pylab

import pywrapfst as openfst


VISUALISE=True

def make_label(path, categories, outfile):
    start = 0.0
    f = open(outfile, 'w')
    for cat, dur in path:
        end = (start + dur) / 100.0 ## 10msec -> sec
        f.write('%s %s %s\n'%(start / 100.0, end, categories[cat]))
        start += dur
    f.close()



def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-f', dest='feat_dir', required=True)
    a.add_argument('-l', dest='lab_dir', required=True)
    a.add_argument('-o', dest='outdir', required=True)        
    a.add_argument('-s', dest='smooth_length', required=False, default=600, type=int, help='Length of smoothing window (in frames) for class posteriors')
    opts = a.parse_args()
    
    # ===============================================

    # feat_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/bbc_data_chunking/hausa/mfcc/'
    # lab_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/bbc_data_chunking/hausa/lab/'
    # outdir = '/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/bbc_data_chunking/hausa/out'

    feat_dir = opts.feat_dir
    lab_dir = opts.lab_dir
    outdir = opts.outdir
    smooth_length = opts.smooth_length

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    categories = ['s','ms','m','p']
    categories_full = ['speech','speech & music','music','report']

    features, labels, unlabelled, unlabelled_names = load_data(feat_dir, lab_dir, categories)

    print 'build forest of trees....'
    from sklearn.ensemble import RandomForestClassifier
    
    clf = RandomForestClassifier(max_depth=10, random_state=0)
    clf.fit(features, labels)
    print clf

    print 'apply to new data'

    nepisodes = len(unlabelled) # to debug, set to e.g.: 2

    if VISUALISE:
        assert nepisodes < 10
    for episode in range(nepisodes):
        print '--- episode %s ---'%(episode)

        #### Smoothed version:-
        guess, smooth_guess, changepoints = decode_episode(unlabelled[episode], clf, smooth_length, categories, decode_smoothed=True, refine= True)
        outlab = os.path.join(outdir, unlabelled_names[episode] + '.lab')

        #### Unsmoothed version (does not work -- many little s-p pairs detected):-
        #guess, smooth_guess, changepoints = decode_episode(unlabelled[episode], clf, smooth_length, categories, decode_smoothed=False)
        #outlab = os.path.join(outdir, 'lab2', unlabelled_names[episode] + '.lab')


        make_label(changepoints, categories, outlab)

        if VISUALISE:
            pylab.subplot('%s1%s'%(nepisodes, episode+1))
            # guess = clf.predict_proba(unlabelled[episode])

            # m,n = guess.shape

            # filt = np.hanning(smooth_length) / np.hanning(smooth_length).sum()
            # # guess_smooth = np.zeros(guess.shape)

            m,n = guess.shape

            for i in range(n):
                #smoothed = scipy.signal.convolve(guess[:,i], filt, mode='same')
                pylab.plot(smooth_guess[:,i], label=categories_full[i])

            if episode == 0:
                pylab.legend()


    if VISUALISE:
        pylab.show()



def deduplicate_path(seq):
    current = seq[0]
    reduced = []
    count = 0
    for thing in seq:

        if thing != current: # change on label

            # reduced.append((current, count))
            # count = 0
            # current = thing

            if (current != 3) or ( (current == 3) and (count > 1000) ):

                if len(reduced) > 0:
                    ( prev_curr , prev_count ) = reduced[-1]
                else:
                    prev_curr = -1

                if prev_curr == current : # add to previous count
                    reduced.pop()
                    count += prev_count

                reduced.append((current, count))
                count = 0
            # It has to skip
            current = thing

        count += 1

    reduced.append((current, count))
    return reduced

def get_shortest_path(fst):
    shortest_path = openfst.shortestpath(fst, weight=None) 

    ## reverse order when printing -- TODO investigate why and use proper operations to extract path
    data = [line.split('\t') for line in shortest_path.text().split('\n')]
    data = [line for line in data if len(line) in [4, 5]]
    data = [(int(line[0]), int(line[2])) for line in data] # (i,o,lab1,lab2,[weight])
    data.sort()
    data.reverse()

    shortest_path = [cat-1 for (index, cat) in data if cat != 0] ## remove epsilon 0
    #            ^--- back to python indices
    return shortest_path

def make_schema_lattice(categories, mindur=0):
    # episode_start = ['m', 'ms', 'm']   ### start and ends of episodes are fixed...
    # reports = ['s', 'p']                                 ## there can be 1 or more of these
    # episode_end = ['ms', 'm', 's', 'p', 's', 'ms', 'm']

    schema = ['m', 'ms', 'm',     's', 'p',      'ms', 'm', 's', 'p', 's', 'ms', 'm']


    catmap = dict(zip(categories, range(1, len(categories)+1)))
    #durmap = {'m': 100}

    fst = []
    start = 0

    for i in range(len(schema)):

        if i==3:
            reports_start = start + 1

        for st in range(mindur):
            end = start + 1
            state = catmap[schema[i]]
            fst.append('%s %s %s %s'%(start, end, state, state)) 
            start = end

        end = start + 1
        state = catmap[schema[i]]
        fst.append('%s %s %s %s'%(start, end, state, state)) 
        ## add self-loop everywhere:
        fst.append('%s %s %s %s'%(end, end, state, state)) 
        start = end

        if i==4:
            reports_end = end

    ## add loop to repeat reports:
    fst.append('%s %s %s %s'%(reports_end, reports_start,     0, 0)) 

    fst.append('%s'%(end)) 
    
    compiler = openfst.Compiler()    
    for line in fst:
        print >> compiler, line
        #compiler.write(line)   ## TODO test
    f = compiler.compile()
    f.arcsort(st="ilabel")

    f.draw('/tmp/schema.dot')

    return f



def make_sausage_lattice(posteriors):
    '''
    sort on outsymbols
    '''
    posteriors = np.log(posteriors) * -1.0      

    fst = []
    start = 0
    frames, cands = np.shape(posteriors)

    for i in range(frames):
    
        end = start + 1
        for j in range(cands):
            
            state = j+1
            weight = posteriors[i,j] 
            fst.append('%s %s %s %s %s'%(start, end, state, state, weight)) 
            
        start = end
    fst.append('%s'%(end)) 
    
    compiler = openfst.Compiler()    
    for line in fst:
        print >> compiler, line
        #compiler.write(line)   ## TODO test
    f = compiler.compile()
    f.arcsort(st="olabel")
    return f


def smooth_posteriors(posteriors, smooth_length):
    filt = np.hanning(smooth_length) / np.hanning(smooth_length).sum()
    posteriors_smooth = np.zeros(posteriors.shape)
    m,n = posteriors.shape
    for i in range(n):
        smoothed = scipy.signal.convolve(posteriors[:,i], filt, mode='same')
        posteriors_smooth[:,i] = smoothed
    return posteriors_smooth
    

def read_labels(fname, categories):
    catmap  = dict(zip(categories, range(len(categories))))

    f = open(fname, 'r')
    lines = f.readlines()
    f.close()
    lines = [line.strip(' \n').split(' ') for line in lines]
    total_frames = 0
    frames_cats = []
    for (start, end, category) in lines:
        assert category in categories

        seconds = float(end) - float(start)
        frames = int(seconds * 100.0)
        frames_cats.append((frames, catmap[category]))

        total_frames += frames

    data = np.ones(total_frames)
    start = 0
    for (frames, cat) in frames_cats:
        data[start:start+frames] = cat
        start += frames
    return data

def load_data(feat_dir, lab_dir, categories):
    '''
    '''

    assert len(glob.glob(lab_dir + '/*.lab')) > 0, 'no labels in %s'%(lab_dir)

    feats = glob.glob(feat_dir + '/*.mfcc') # [:3]
    #labs = glob.glob(lab_dir + '/*.lab')


    feats_only = []
    unlabelled_names = []
    labelled_feats = []
    all_labels = []

    for feat in feats[:6]:
        _, base = os.path.split(feat)
        base = base.replace('.mfcc','')
        print 'loading %s'%(base)

        lab = os.path.join(lab_dir, base + '.lab')

        features = get_speech(feat, 13, remove_htk_header=True)

        feats_only.append(features)
        unlabelled_names.append(base)

        if not os.path.isfile(lab):
            continue
        
        print 'getting label for %s'%(base)


        labels = read_labels(lab, categories)

        feat_frames,n = features.shape
        lab_frames, = labels.shape

        
        frames = min(feat_frames, lab_frames)
        
        features = features[:frames,:]
        labels = labels[:frames]

        labelled_feats.append(features)
        all_labels.append(labels)

    #feats_only = np.vstack(feats_only)
    labelled_feats = np.vstack(labelled_feats)
    all_labels = np.concatenate(all_labels)

    return labelled_feats, all_labels, feats_only, unlabelled_names

def decode_episode(data, clf, smooth_length, categories, decode_smoothed=True, refine=False):
    guess = clf.predict_proba(data)

    smooth_guess = smooth_posteriors(guess, smooth_length)
    
    schema = make_schema_lattice(categories)

    print schema
    schema.draw('/tmp/schema.dot')
    sys.exit('adcadv')


    if decode_smoothed:
        sausage = make_sausage_lattice(smooth_guess)
    else:
        sausage = make_sausage_lattice(guess)

    composed_latt = openfst.compose(sausage, schema) 
    shortest_path = get_shortest_path(composed_latt)

    s = deduplicate_path(shortest_path)

    if refine and decode_smoothed :
        s = refine_changepoints(s, guess, schema)

    # if refine and decode_smoothed:
    #     propr = np.array([0.05]) # % of original smooth window
    #     for p in range(len(propr)):
    #         less_smooth_length = int(smooth_length * propr[p])
    #         less_smooth_guess = smooth_posteriors(guess, less_smooth_length)
    #         s = refine_changepoints_guess(smooth_guess, less_smooth_guess, s, less_smooth_length)

    print s

    # print shortest_path

    for cat, dur in s:
        print '%s      %s'%(categories[cat], dur)

    return guess, smooth_guess, s        

def refine_changepoints(s, guess, schema):

    # Find less smooth changepoints
    less_smooth_guess   = smooth_posteriors(guess, 100)
    less_smooth_sausage = make_sausage_lattice(less_smooth_guess)
    composed_latt       = openfst.compose(less_smooth_sausage, schema)
    shortest_path       = get_shortest_path(composed_latt)
    less_smooth_s       = deduplicate_path(shortest_path)

    # Move changepoint to closest less smoothed one
    ind = 0
    current = 0
    smooth_cp = np.zeros(len(s))
    smooth_ct = np.zeros(len(s))
    for cat, dur in s:
        current += dur
        smooth_cp[ind] = current # changepoint
        smooth_ct[ind] = cat
        ind +=1

    ind = 0
    current = 0
    less_smooth_cp = np.zeros(len(less_smooth_s))
    less_smooth_ct = np.zeros(len(less_smooth_s))
    for cat, dur in less_smooth_s:
        current += dur
        less_smooth_cp[ind] = current # changepoint
        less_smooth_ct[ind] = cat
        ind +=1

    print smooth_cp
    print less_smooth_cp

    ind = 0
    new_changepoints = []
    for cat, dur in s:
        # Find closest change point
        tmp  = np.argmin( abs(smooth_cp[ind] - less_smooth_cp) )
        diff = abs(smooth_cp[ind] - less_smooth_cp[tmp])
        if diff < 100:
            print "Replace this " + str(int(smooth_cp[ind])) + " by this: " + str(int(less_smooth_cp[tmp]))
            smooth_cp[ind] = less_smooth_cp[tmp]

        if ind > 0 :
            new_dur = smooth_cp[ind] - smooth_cp[ind-1]
        else:
            new_dur = smooth_cp[ind]
        new_changepoints.append( ( int(smooth_ct[ind]), int(new_dur) ) )
        ind +=1

    return new_changepoints

# doesn't work well
def refine_changepoints_guess(guess, rough_guess, changepoints, less_smooth_length):

    win_size = 5*less_smooth_length # in frames
    win = np.hanning(win_size*4)

    start = 0
    new_changepoints = []

    for cat, dur in changepoints:

        cp = start # changepoint

        if cp != 0 :

            win_range = np.arange(cp-win_size , cp+win_size)

            # Find probability values of left hand size category (prev cat)
            pl = rough_guess[win_range, prev_cat]

            # Find probability values of right hand size category (current cat)
            pr = rough_guess[win_range, cat]

            # Apply hanning window to it
            pl = win[:2*win_size] * pl # first half of win
            pr = win[2*win_size:] * pr # second half of win

            # Find the point where pl and pr are closest
            tmp = np.argmin( abs(pl - pr) )
            if tmp < win_size:
                new_cp = cp - (win_size - tmp)
            else:
                new_cp = cp + (tmp - win_size)

            print str(cp) + ' ' + str(new_cp)

            new_dur = new_cp - prev_cp
            if new_dur > 0 :
                new_changepoints.append( (prev_cat, new_dur) )

        else:

            new_cp = cp

        start    += dur
        prev_cat = cat
        prev_cp  = new_cp

    new_changepoints.append( (cat, guess.shape[0] - new_cp ) )

    return new_changepoints

if __name__=="__main__":
    #deduplicate_path([1,2,4,4,5,2,2,1])
    main_work()

