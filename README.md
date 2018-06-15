# Audioseg

These are scripts which implement a simple approach to segmenting the audio of broadcast news episodes into chunks. 

## Installation of Python dependencies with virtual environment

The scripts depend on various Python packages. The best way to install these is with pip in a virtual environment. Make a directory to house virtual environments if you don't already have one, and move to it:

```
cd /path/to/virtual_python/
virtualenv --distribute --python=/usr/bin/python2.7 audioseg
source ./audioseg/bin/activate
```

Now install the requirements by moving to your copy of the repository and doing:

```
pip install -r ./requirements.txt
```



## Extracting wavfiles from data

The segmentation scripts work with .wav format audio files at 16kHz. Assuming the data we are working with is in one or more of the formats mp3, mp4, m4a, or are wav files at a different sample rate, use the script ```./script/get_waves.sh``` to extract audio in the right format. Set a variable to point to the folder containing this data:

```
ORIG_DATA_DIR=/path/to/your/MP3s/
```

Set a location to work in:

```
WORKDIR=/path/to/output/
```

Extract 16kHz and 48kHz (or 44.1kHz, depending on the format of the data) waveforms from both waveforms and videos in `$ORIG_DATA_DIR` (note that only the 16kHz versions are used here):

```
./script/get_waves.sh $ORIG_DATA_DIR $WORKDIR/wav16 $WORKDIR/wav48
```

This script requires the tools `sox` and `ffmpeg` to be installed and on your path.



## Audio segmentation

### Initial label(s)

Use e.g. wavesurfer (http://www.speech.kth.se/wavesurfer/) to produce a label file for 1 episode (or more, but a single 10 minute episode seems to be enough). Place the label, named consistently with the wav file (replacing extension .wav with .lab), inside `$WORKDIR/lab/`. The label should mark a number of categories on the basis of which you wish to segment the audio. No particular categories are predefined; previously we have used:

- m: music only
- s: speech from target presenter
- ms: speech from target presenter with music in background
- p: audio from package (pkg)

Alternatively, copy a label file that has already been made by hand for an episode:

```
mkdir $WORKDIR/lab/
cp ./egs/manual_label/hausa/hausa_bulletin_2017_1_27.lab $WORKDIR/lab/
```

This manually produced label contains the following data:

```
0.0000000 19.7126313 m
19.7126313 51.6338517 ms
51.6338517 55.8505194 m
55.8505194 85.3678154 s
85.3678154 257.0338855 p
257.0338855 270.4477203 s
270.4477203 292.3518146 ms
292.3518146 295.7246574 m
295.7246574 318.0164348 ms
318.0164348 320.6139115 m
320.6139115 342.9444572 ms
342.9444572 348.9923133 m
348.9923133 375.8587512 s
375.8587512 504.6083038 p
504.6083038 581.7960060 s
581.7960060 593.4264986 ms
593.4264986 602.3432096 m
```


### Extract frame-level probabilities

Now run the following command to extract acoustic features at 10msec intervals, train a predictor for the classes you defined on the labelled file, and then to output framewise class probabilities for all episodes (labelled and unlabelled):

```
python script/get_class_probabilities.py -w $WORKDIR/wav16/ -l $WORKDIR/lab/ -o $WORKDIR/classprobs/ -s 600
```

Frame-level class probabilities are output in 2 forms (smoothed and unsmoothed) in `$WORKDIR/classprobs/`. The smoothed ones end `_smoothed.csv`.  Option `-s` controls how class probabilities are smoothed. 


### Visualise frame-level probabilities

```
python script/plot_class_probabilities.py -i $WORKDIR/classprobs/
```

![Plot of class probabilities](https://github.com/oliverwatts/audioseg/blob/master/egs/plots/class_probs.png)

### Decode episdoes

TODO


