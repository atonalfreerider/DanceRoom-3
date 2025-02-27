#!/bin/bash

# Parse named arguments
OPTIONS=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --options=*) OPTIONS="${1#*=}"; shift ;;
        *) break ;;
    esac
done

# Assign remaining arguments to variables
INPUT_VIDEO=$1
OUTPUT_DIR=$2
DPVO_PATH=/home/john/Desktop/Video/DPVO
BEAT_THIS_PATH=/home/john/Desktop/Audio/beat_this
NLF_PATH=/home/john/Desktop/3DPose/nlf

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Initialize conda
eval "$(conda shell.bash hook)"

# Run DPVO with options if specified
conda activate dpvo
cd "$DPVO_PATH"
DPVO_ARGS=""
python demo.py --video_path=$INPUT_VIDEO

# Move DPVO output to output directory
mv "${INPUT_VIDEO%.*}.vo.json" "$OUTPUT_DIR"

# Run Beat_This
conda activate beat_this
cd $BEAT_THIS_PATH
python beat_this_analyzer.py $INPUT_VIDEO
# Move Beat_This output to output directory
mv ${INPUT_VIDEO%.*}_zouk-time-analysis.json $OUTPUT_DIR/zouk-time-analysis.json

# Run NLF
conda activate nlf
cd $NLF_PATH
python main.py $INPUT_VIDEO $OUTPUT_DIR

# Separate Audio Data
ffmpeg -y -i $INPUT_VIDEO -vn -acodec pcm_s16le $OUTPUT_DIR/audio.wav

# Run DanceRoom3 scripts
conda activate DanceRoom3
cd /home/john/Desktop/3DPose/DanceRoom-3
python pose_tracker3d.py --output_dir=$OUTPUT_DIR
python sam2.py --video=$INPUT_VIDEO --output_dir=$OUTPUT_DIR

# Run Video-Depth-Anything
conda activate VideoDepthAnything
cd /home/john/Desktop/Video/Video-Depth-Anything
python run.py --input_video $INPUT_VIDEO --output_dir $OUTPUT_DIR

echo "Processing complete. Outputs are in $OUTPUT_DIR"