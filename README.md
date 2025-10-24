# DanceRoom-3

## INSTALLATION  

conda env create -f environment.yml

## RUN SHELL  

run the process_video.sh
- provide codebase locations

process_video.sh <input-video-path> <output-dir-path>  

## TRACK PREPROCESS  

Run Human3r to generate 3D poses:
The Human3r output should be saved as: <video-name>-poses.json

Expected format:
```json
{
  "metadata": { ... },
  "frames": {
    "0": {
      "camera": { ... },
      "humans": [
        {
          "person_id": 0,
          "smpl_parameters": { ... }
        }
      ]
    }
  }
}
```

Run Beat_This:  
https://github.com/atonalfreerider/beat_this

Beat_This model:  
https://cloud.cp.jku.at/index.php/s/7ik4RrBKTS273gp/download?path=%2F&files=final0.ckpt

conda activate beat_this  
python ...path-to/beat_this/beat_this_analyzer.py <input-video-path>

Output (to same location as <input-video-path>):
- <video-name>_zouk-time-analysis.json

Separate Audio Data:
ffmpeg -i /path/to/vid.mp4 -vn -acodec pcm_s16le path/to/out/audio.wav

## RUN

conda activate DanceRoom3  
python pose_tracker3d.py --output_dir=/path
python video_meta.py <path_to_video.mp4> <output_dir>

Output figure1.json and figure2.json to be provided to HeadMovement, with audio, video meta, and zouk beat
https://github.com/atonalfreerider/head-movement


