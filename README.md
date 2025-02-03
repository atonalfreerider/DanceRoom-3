# DanceRoom-3

## INSTALLATION  

conda env create -f environment.yml

## RUN SHELL  

run the process_video.sh
- provide codebase locations

prochess_video.sh <input-video-path> <output-dir-path>  

## TRACK PREPROCESS  

Run DPVO:  
https://github.com/atonalfreerider/DPVO  
DPVO model:  
https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip

conda activate dpvo  
...path-to/DPVO/demo.py --video_path=<intput-video-path>  

Output (to same location as <intput-video-path>):  
- visual odometry json, where zoom and Z are conflated

Run Beat_This:  
https://github.com/atonalfreerider/beat_this

Beat_This model:  
https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp/download?path=%2F&files=final0.ckpt

conda activate beat_this  
python ...path-to/beat_this/beat_this_analyzer.py <intput-video-path>

Output (to same location as <intput-video-path>):
- <video-name>_zouk-time-analysis.json

Run NLF:
https://github.com/atonalfreerider/nlf

NLF model:
https://github.com/isarandi/nlf/releases/download/v0.2.0/nlf_l_multi.torchscript

conda activate nlf  
python ...path-to/nlf/main.py <input-video-path> <path-to-out-dir>  

Separate Audio Data:
ffmpeg -i /path/to/vid.mp4 -vn -acodec pcm_s16le path/to/out/audio.wav

## RUN

conda activate DanceRoom3  
python poses_tracker3d.py --output_dir=/path

python sam2.py --video=/path --json_folder=/path --output_dir=/path

python video_meta.py <path_to_video.mp4>"

output figure1.json and figure2.json to be provided to HeadMovement, with audio, video meta, and zouk beat
https://github.com/atonalfreerider/head-movement


