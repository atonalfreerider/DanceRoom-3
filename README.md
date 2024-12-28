# DanceRoom-3

## INSTALLATION  

conda env create -f environment.yml

## TRACK PREPROCESS  

Run DPVO:  
https://github.com/atonalfreerider/DPVO  
DPVO model:  
https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip

Output:
- visual odometry json, where zoom and Z are conflated

Run Beat_This:  
https://github.com/atonalfreerider/beat_this

Beat_This model:  
https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp/download?path=%2F&files=final0.ckpt

Output:
- rhythm json

Run NLF:
https://github.com/atonalfreerider/nlf

NLF model:
https://github.com/isarandi/nlf/releases/download/v0.2.0/nlf_l_multi.torchscript

Separate Audio Data:
ffmpeg -i /path/to/vid.mp4 -vn -acodec pcm_s16le path/to/out/audio.wav

## RUN

python poses_tracker3d.py --output_dir=/path

output figure1.json and figure2.json to be provided to HeadMovement and poses2d_boxes
https://github.com/atonalfreerider/head-movement

python sam2.py --video=/path --json_folder=/path --output_dir=/path

python video_meta.py <path_to_video.mp4>"





