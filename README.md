# M2G Pipeline - Motion to Music Generation for Figure Skating

An end-to-end pipeline that analyzes figure skating performances from video, classifies skating moves using Graph Convolutional Networks, and generates synchronized music for each detected move using AI.

## Overview

The M2G (Motion to Music Generation) pipeline transforms figure skating videos into custom AI-generated soundtracks by:

1. **Extracting skeleton poses** from video using OpenPose/MMPose
2. **Classifying skating moves** using a trained GCN model (MDRGCN)
3. **Generating tailored music** for each move using InspireMusic
4. **Concatenating segments** into a cohesive performance soundtrack

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Video     │───▶│  Skeleton   │───▶│    Move     │───▶│   Music     │
│   Input     │    │  Extraction │    │  Prediction │    │ Generation  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                  │                  │
                          ▼                  ▼                  ▼
                   skeleton.npy      moves.json         segments/*.wav
                                                               │
                                                               ▼
                                                    ┌─────────────────┐
                                                    │  Concatenation  │
                                                    │  + Crossfade    │
                                                    └────────┬────────┘
                                                             │
                                                             ▼
                                                   performance_music.wav
```

## Features

- **Automatic Move Detection**: Classifies 10 different skating moves including jumps, spins, and sequences
- **Intelligent Music Mapping**: Each move type has a carefully crafted music prompt
- **Smooth Transitions**: Configurable crossfade between music segments
- **Multiple Output Formats**: WAV (lossless) and MP3 (320kbps)
- **Flexible Entry Points**: Start from video, skeleton data, or pre-classified moves
- **Interactive Mode**: User-friendly command-line interface

## Supported Skating Moves

| Move | Description | Music Style |
|------|-------------|-------------|
| `2Axel` | Double Axel jump | Dramatic orchestral with brass crescendo |
| `3Axel` | Triple Axel jump | Epic cinematic with triumphant fanfare |
| `3Loop` | Triple Loop jump | Graceful waltz-like classical |
| `3Flip` | Triple Flip jump | Dynamic with dramatic accents |
| `3Lutz` | Triple Lutz jump | Powerful symphonic with percussion |
| `3Lutz_3Toeloop` | Combination jump | Explosive dual climactic moments |
| `FlyCamelSpin4` | Flying Camel Spin (Level 4) | Ethereal with spinning motifs |
| `ChComboSpin4` | Combination Spin (Level 4) | Intricate classical with layers |
| `StepSequence3` | Step Sequence (Level 3) | Rhythmic with footwork patterns |
| `ChoreoSequence1` | Choreographic Sequence | Expressive emotional melody |

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg (for audio processing)

### Clone Repository

```bash
git clone https://github.com/yourusername/m2g-pipeline.git
cd m2g-pipeline
```

### Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install numpy pydub

# For skeleton extraction (choose one)
# Option 1: OpenPose
# Follow instructions at https://github.com/CMU-Perceptual-Computing-Lab/openpose

# Option 2: MMPose
pip install openmim
mim install mmcv mmpose

# For music generation
# Clone and setup InspireMusic
git clone https://github.com/FunAudioLLM/InspireMusic.git
cd InspireMusic
pip install -e .
```

### Download Models

```bash
# GCN Model for move classification
mkdir -p model
# Download runs-215-21930.pt to model/

# InspireMusic model (auto-downloads on first run)
# Or manually download:
python -c "from modelscope import snapshot_download; snapshot_download('iic/InspireMusic-1.5B-Long', local_dir='pretrained_models/InspireMusic-1.5B-Long')"
```

## Usage

### Quick Start

```bash
# Interactive mode (recommended for first-time users)
python m2g_pipeline.py --interactive

# Or use the simplified music generator
python generate_music.py --interactive
```

### Full Pipeline (from Video)

```bash
python m2g_pipeline.py \
    --video skating_performance.mp4 \
    --output ./output \
    --gpu 0
```

### From Skeleton Data

If you already have extracted skeleton data:

```bash
python m2g_pipeline.py \
    --skeleton skeleton_full.npy \
    --output ./output
```

### Music Generation Only

If you have pre-classified moves:

```bash
python m2g_pipeline.py \
    --moves moves_prediction.json \
    --output ./output

# Or use the standalone generator
python generate_music.py \
    --moves moves_prediction.json \
    --output ./output \
    --name my_performance
```

### Command Line Options

#### m2g_pipeline.py

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--video` | `-v` | Input video file path | - |
| `--skeleton` | `-s` | Skeleton .npy file path | - |
| `--moves` | `-m` | Moves JSON file path | - |
| `--output` | `-o` | Output directory | `./m2g_output` |
| `--gpu` | `-g` | GPU ID (-1 for CPU) | `0` |
| `--interactive` | `-i` | Interactive mode | `False` |
| `--model-path` | - | GCN model path | `./model/runs-215-21930.pt` |
| `--inspiremusic-dir` | - | InspireMusic model directory | Auto |

#### generate_music.py

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--moves` | `-m` | Moves JSON file path | - |
| `--output` | `-o` | Output directory | `./m2g_output` |
| `--name` | `-n` | Output filename | `performance_music` |
| `--gpu` | `-g` | GPU ID | `0` |
| `--crossfade` | - | Crossfade duration (ms) | `1000` |
| `--interactive` | `-i` | Interactive mode | `False` |

## Input/Output Formats

### Skeleton Data Format

NumPy array with shape `(C, T, V, M)`:
- `C`: Channels (3 - x, y, confidence)
- `T`: Time frames
- `V`: Vertices (25 OpenPose keypoints)
- `M`: Number of persons (1 for single skater)

### Moves JSON Format

```json
[
    {
        "move": "3Lutz_3Toeloop",
        "start_time": 0.0,
        "end_time": 16.67,
        "duration": 16.67,
        "confidence": 0.9234
    },
    {
        "move": "StepSequence3",
        "start_time": 16.67,
        "end_time": 33.33,
        "duration": 16.66,
        "confidence": 0.8756
    }
]
```

### Output Structure

```
m2g_output/
├── skeletons/                    # Step 1 output
│   ├── json/                     # OpenPose JSON files
│   └── skeleton_full.npy         # Consolidated skeleton data
├── predictions/                  # Step 2 output
│   └── moves_prediction.json     # Classified moves
├── music_segments/               # Step 3 output
│   ├── seg_000_3Lutz_3Toeloop.wav
│   ├── seg_001_StepSequence3.wav
│   ├── ...
│   └── generation_log.json
└── final/                        # Step 4 output
    ├── performance_music.wav     # Final concatenated audio
    ├── performance_music.mp3     # MP3 version
    └── pipeline_results.json     # Complete results log
```

## Configuration

### Customizing Music Prompts

Edit the `MOVE_PROMPTS` dictionary in either script to customize music generation:

```python
MOVE_PROMPTS = {
    "3Axel": {
        "prompt": "Your custom prompt here",
        "chorus": "chorus",  # intro, verse, chorus, outro
        "intensity": "maximum"
    },
    # ... other moves
}
```

### Pipeline Configuration

```python
@dataclass
class PipelineConfig:
    # Video processing
    video_fps: int = 30
    
    # Move prediction
    model_window: int = 500      # Frames per prediction window
    interval: int = 500          # Step size between windows
    min_frames: int = 300        # Minimum frames for valid chunk
    
    # Music generation
    min_music_duration: float = 5.0
    max_music_duration: float = 30.0
    crossfade_duration: int = 1000  # milliseconds
    output_sample_rate: int = 48000
```

## API Reference

### M2GPipeline

```python
from m2g_pipeline import M2GPipeline, PipelineConfig

# Initialize
config = PipelineConfig(output_dir="./output", gpu_id=0)
pipeline = M2GPipeline(config)

# Run full pipeline
results = pipeline.run(
    video_path="skating.mp4",      # Start from video
    # skeleton_path="skeleton.npy",  # Or from skeleton
    # moves_json="moves.json"        # Or from moves
)

print(results["outputs"]["final_audio"])
```

### MusicPipelineGenerator

```python
from generate_music import MusicPipelineGenerator

# Initialize
generator = MusicPipelineGenerator(
    output_dir="./output",
    gpu=0,
    crossfade_ms=1000
)

# Generate music
results = generator.run(
    moves_path="moves_prediction.json",
    output_name="my_performance"
)

print(results["output"]["wav_path"])
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size or use CPU
python m2g_pipeline.py --gpu -1 ...
```

**2. OpenPose Not Found**
```bash
# Set the correct path
export OPENPOSE_DIR=/path/to/openpose
python m2g_pipeline.py --video input.mp4 ...
```

**3. InspireMusic Download Failed**
```bash
# Manually download the model
python -c "
from modelscope import snapshot_download
snapshot_download('iic/InspireMusic-1.5B-Long', 
                  local_dir='pretrained_models/InspireMusic-1.5B-Long')
"
```

**4. Audio Concatenation Issues**
```bash
# Install FFmpeg
sudo apt-get install ffmpeg

# Or on macOS
brew install ffmpeg
```

### Performance Tips

- Use GPU for both move prediction and music generation
- For long videos, consider processing in chunks
- Pre-extract skeletons to iterate on music generation faster
- Adjust `crossfade_duration` for smoother/sharper transitions

## Project Structure

```
m2g-pipeline/
├── m2g_pipeline.py          # Main unified pipeline
├── generate_music.py        # Standalone music generator
├── sample_moves.json        # Example moves file
├── README.md
├── requirements.txt
├── model/
│   ├── MDRGCN.py            # GCN model architecture
│   └── runs-215-21930.pt    # Trained weights
├── graph/
│   └── skating.py           # Skeleton graph definition
├── inspiremusic/            # InspireMusic submodule
│   └── cli/
│       └── inference.py
└── pretrained_models/
    └── InspireMusic-1.5B-Long/
```

