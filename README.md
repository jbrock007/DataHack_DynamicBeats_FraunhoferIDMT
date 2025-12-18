# Data Hack - FraunhoferIDMT_DynamicBeats

This repository provides a pipeline to predict figure skating moves from pre-extracted skeleton data(https://github.com/CMU-Perceptual-Computing-Lab/openpose) using a pretrained MDR-GCN model(https://github.com/dingyn-Reno/MDR-GCN). The pipeline outputs time-stamped moves in JSON format. Than integrated them with prompt to generate music using InspireMusic(https://github.com/FunAudioLLM/FunMusic) that syncs with video.

---

## Repository Structure

```plaintext
repo/
├── data/                # Pre-extracted skeleton data (.npy files)            
├── model/               # Pretrained model file (.pt)
├── outputs/             # JSON outputs from predictions     
├── src/                 # Python scripts
├──  └── inference.py    # Run predictions 
└── README.md            # Project documentation
```

---

## Features

- Loades keypoints extracted from video for each frame from data/ folder.
- Filter to get keypoints of only perfomer.
- Post-processing skeleton to feed into model for prediction.
- Predicts move by feeding model chunks of 500 frames.
- Outputs .json file saved in output/ folder with moves predicted for each chunk of frames.

- The part which calculates the skeleton data using Openpose and post-processing of those keypoints works independently from this repo, it would be integrated and add to this repo soon.

### 2. Music Prompt Generation
- Analyzes predicted moves and their timestamps
- Generates contextual music prompts based on move characteristics:
  - **Jumps (Axel, Lutz, Flip)**: Dramatic, powerful orchestration with brass accents
  - **Spins (Camel, Flying, Sit)**: Graceful, flowing melodies with strings
  - **Step Sequences**: Energetic, rhythmic passages
  - **Transitions**: Smooth tempo changes and dynamic variations
- Creates segmented prompts for continuous music generation (up to 45 seconds per segment)
- Outputs both full program prompts and segment-specific prompts

### 3. Music Generation with InspireMusic
- Integrates with InspireMusic AI model for music generation
- Supports both text-to-music and continuation modes
- Generates music in segments that seamlessly connect
- Chains segments together for full program duration (up to 2+ minutes)
- Outputs high-quality synchronized audio files in WAV format (48kHz stereo)
---

## How to Use

# 1. Clone the repository

```bash
git clone https://github.com/yourusername/SkatingMovePrediction.git

cd SkatingMovePrediction
```

# 2. Run the inference script

```bash
python src/inference.py
```
