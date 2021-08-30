   
import os
import sys
import glob

####################################################################
# FFMPEG Installation
# Download and unzip files 
# https://www.ffmpeg.org/download.html
####################################################################

sys.path.append(r'C:\ffmpeg\bin') # Path to exe files - ffmpeg, ffplay, ffprobe --> User specific

def duplicate_remover(video, output):
    os.system('ffmpeg -i "{video}" -vf mpdecimate,setpts=N/FRAME_RATE/TB "{output}"'.format(video=video, output=output))

#####################################################################   
# -i --> input file
# -vf --> video filter
# mpdecimate --> https://ffmpeg.org/ffmpeg-all.html#toc-mpdecimate
# setpts --> set presentation time stamps
#more info: https://stackoverflow.com/questions/43333542/what-is-video-timescale-timebase-or-timestamp-in-ffmpeg/43337235#43337235
#####################################################################

if __name__ == '__main__':
    videos_path = input('Copy and paste the path of videos: ')
    os.chdir(videos_path)
    path = os.path.join(videos_path,'Cleaned')
    try:
        os.mkdir(path)
    except:
        pass
    videos_list = [f for f in glob.glob('*.mp4')]
    print('Total number of videos found: ', len(videos_list))

    for file in videos_list:
        try:
            duplicate_remover(file, os.path.join(path,'Cleaned '+file))
        except:
            print('Error processing the file:',file)
    
    
    