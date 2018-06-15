
INDIR=$1       # input directory that can contain wav, mp4, mp4 or m4a media files with stereo audio
OUTDIR16=$2    # directory to store wavefiles 16kHz sampling frequency (one channel)
OUTDIRHIGH=$3  # directory to store wavefiles with the original higher sampling frequency (one channel)

mkdir -p $OUTDIR16
mkdir -p $OUTDIRHIGH

# Done in this order in case the same program is available in different formats. 
# For Swahili there is an overlap between video and audio: ( > means better )
# mp3 (mp3 codec 256kb/s) > mp4 (aac codec 125kb/s) because of the bit rate

for VID in $INDIR/*mp4 ; do
    BASE=`basename $VID .mp4` ;
    echo $BASE ;
    ffmpeg -i $VID /tmp/$BASE.wav ;
    sox /tmp/$BASE.wav $OUTDIRHIGH/$BASE.wav remix 1; 
    sox $OUTDIRHIGH/$BASE.wav -r 16000 $OUTDIR16/$BASE.wav ; 
    rm /tmp/$BASE.wav ;
done

for WAVE in $INDIR/*mp3 ; do
    BASE=`basename $WAVE .mp3` ;
    echo $BASE ;
    ffmpeg -i $WAVE /tmp/$BASE.wav ;
    sox /tmp/$BASE.wav $OUTDIRHIGH/$BASE.wav remix 1; 
    sox $OUTDIRHIGH/$BASE.wav -r 16000 $OUTDIR16/$BASE.wav ;
    rm /tmp/$BASE.wav ;
done

for WAVE in $INDIR/*m4a ; do
    BASE=`basename $WAVE .m4a` ;
    echo $BASE ;
    ffmpeg -i $WAVE /tmp/$BASE.wav ;
    sox /tmp/$BASE.wav $OUTDIRHIGH/$BASE.wav remix 1; 
    sox $OUTDIRHIGH/$BASE.wav -r 16000 $OUTDIR16/$BASE.wav ; 
    rm /tmp/$BASE.wav ;
done

for WAVE in $INDIR/*wav ; do
    BASE=`basename $WAVE .wav` ;
    echo $BASE ;
    sox $WAVE $OUTDIRHIGH/$BASE.wav remix 1;
    sox $OUTDIRHIGH/$BASE.wav -r 16000 $OUTDIR16/$BASE.wav ; 
done
