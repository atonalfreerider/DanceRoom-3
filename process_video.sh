#!/bin/bash

# Enable error reporting
set -e

# Parse named arguments
CAMERA_MODEL=""
FIXED_FOCAL_LENGTH=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --camera_model=*) CAMERA_MODEL="${1#*=}"; shift ;;
        --fixed_focal_length=*) FIXED_FOCAL_LENGTH="${1#*=}"; shift ;;
        --fixed_focal_length) FIXED_FOCAL_LENGTH="true"; shift ;;
        *) break ;;
    esac
done

# Assign remaining arguments to variables
INPUT_VIDEO=$1
OUTPUT_DIR=$2
BEAT_THIS_PATH=/home/john/Desktop/Audio/beat_this

# Validate camera model
if [ -z "$CAMERA_MODEL" ]; then
    echo "ERROR: --camera_model is required (static, tripod, or handheld)"
    exit 1
fi

if [[ ! "$CAMERA_MODEL" =~ ^(static|tripod|handheld)$ ]]; then
    echo "ERROR: Invalid camera model '$CAMERA_MODEL'. Must be: static, tripod, or handheld"
    exit 1
fi

# Validate and normalize fixed focal length
if [ -z "$FIXED_FOCAL_LENGTH" ]; then
    FIXED_FOCAL_LENGTH="false"
elif [[ "$FIXED_FOCAL_LENGTH" =~ ^(true|True|TRUE|1)$ ]]; then
    FIXED_FOCAL_LENGTH="true"
else
    FIXED_FOCAL_LENGTH="false"
fi

# Validate input video exists
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "ERROR: Input video not found: $INPUT_VIDEO"
    exit 1
fi

# Validate output directory is specified
if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory not specified"
    exit 1
fi

# Create output directory (ignore if exists, -p flag handles this)
mkdir -p "$OUTPUT_DIR"

# Initialize conda
eval "$(conda shell.bash hook)"

# Extract video base name for Human3r JSON
VIDEO_NAME=$(basename "${INPUT_VIDEO%.*}")
HUMAN3R_JSON="${INPUT_VIDEO%.*}-poses.json"

# Check if Human3r JSON exists
if [ ! -f "$HUMAN3R_JSON" ]; then
    echo "========================================"
    echo "ERROR: Human3r poses file not found!"
    echo "========================================"
    echo "Looking for: $HUMAN3R_JSON"
    echo "Please run Human3r on the video first to generate the poses file."
    exit 1
fi

echo "========================================"
echo "Found Human3r poses file: $HUMAN3R_JSON"
echo "========================================"

# Run Beat_This
echo ""
echo "========================================"
echo "Running Beat_This analysis..."
echo "========================================"
conda activate beat_this
cd $BEAT_THIS_PATH
python beat_this_analyzer.py "$INPUT_VIDEO"

# Move Beat_This output to output directory (overwrite if exists)
mv -f "${INPUT_VIDEO%.*}_zouk-time-analysis.json" "$OUTPUT_DIR/zouk-time-analysis.json"
echo "✓ Beat analysis complete"

# Separate Audio Data
echo ""
echo "========================================"
echo "Extracting audio..."
echo "========================================"
ffmpeg -y -i "$INPUT_VIDEO" -vn -acodec pcm_s16le "$OUTPUT_DIR/audio.wav" 2>&1 | grep -E "(Duration|size=)"
echo "✓ Audio extraction complete"

# Run DanceRoom3 scripts
echo ""
echo "========================================"
echo "Running pose tracker (world space)..."
echo "========================================"
conda activate DanceRoom3
cd /home/john/Desktop/3DPose/DanceRoom-3
python pose_tracker3d.py --poses_json="$HUMAN3R_JSON" --camera_model="$CAMERA_MODEL" --fixed_focal_length="$FIXED_FOCAL_LENGTH" --output_dir="$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Extracting video metadata..."
echo "========================================"
python video_meta.py "$INPUT_VIDEO" "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "✓ Processing complete!"
echo "========================================"
echo "Outputs are in: $OUTPUT_DIR"
echo "Camera model used: $CAMERA_MODEL"
echo "Fixed focal length: $FIXED_FOCAL_LENGTH"