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

import matplotlib  
matplotlib.use('TkAgg') 
import pylab

def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-i', dest='indir', required=True)
    # a.add_argument('-o', dest='outfile', required=True)        
    opts = a.parse_args()
    
    # ===============================================

    flist = sorted(glob.glob(opts.indir + '/*_smoothed.csv'))
    flist = flist[:min(10,len(flist))] ## take 9 at most
    nepisodes = len(flist)
    for (i,csv) in enumerate(flist):
    
        _, episode_name = os.path.split(csv)
        episode_name = episode_name.replace('_smoothed.csv', '')
        data = np.loadtxt(csv)
        if i == 0:
            f = open(csv, 'r')
            header = f.readline()
            f.close()
            header = header.strip(' #\n').split(' ')
            
        ax = pylab.subplot('%s1%s'%(nepisodes, i+1))
       

        m,n = data.shape

        for column in range(n):
            ax.plot(data[:,column], label=header[column])
        ax.set_title(episode_name)
        if i == 0:
            pylab.legend()

    pylab.show()
    # pylab.savefig(opts.outfile)
    # print 'Wrote ' + opts.outfile

if __name__=="__main__":
    main_work()

