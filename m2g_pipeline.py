#!/usr/bin/env python3
"""
M2G (Motion to Music Generation) Unified Pipeline
==================================================
This script provides an end-to-end pipeline for:
1. Analyzing figure skating video to extract skeleton poses
2. Classifying skating moves using GCN model
3. Generating appropriate music for each detected move
4. Concatenating all music segments into a final performance track

"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

# Audio processing
try:
    from pydub import AudioSegment
except ImportError:
    print("Installing pydub...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pydub", "--break-system-packages", "--quiet"])
    from pydub import AudioSegment

try:
    import torch
except ImportError:
    print("PyTorch is required. Please install it first.")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the M2G pipeline"""
    # Paths
    video_path: str = ""
    output_dir: str = "./m2g_output"
    model_path: str = "./model/runs-215-21930.pt"
    inspiremusic_model: str = "InspireMusic-1.5B-Long"
    inspiremusic_dir: str = "./pretrained_models/InspireMusic-1.5B-Long"
    
    # Video processing
    video_fps: int = 30
    
    # Skeleton extraction
    openpose_dir: str = "/usr/local/openpose"
    
    # Move prediction
    model_window: int = 500
    interval: int = 500
    min_frames: int = 300
    
    # Music generation
    min_music_duration: float = 5.0
    max_music_duration: float = 30.0
    crossfade_duration: int = 1000  # milliseconds
    output_sample_rate: int = 48000
    
    # GPU
    gpu_id: int = 0


# =============================================================================
# MOVE TO MUSIC PROMPT MAPPING
# =============================================================================

MOVE_PROMPTS = {
    "2Axel": {
        "prompt": "Dramatic orchestral music with building tension, powerful brass crescendo, elegant and athletic energy, figure skating performance",
        "chorus": "chorus",
        "intensity": "high"
    },
    "3Axel": {
        "prompt": "Epic cinematic orchestral score with soaring strings, triumphant brass fanfare, maximum intensity and grandeur, championship figure skating moment",
        "chorus": "chorus",
        "intensity": "maximum"
    },
    "3Loop": {
        "prompt": "Graceful flowing classical music with swirling strings, elegant waltz-like melody, refined and sophisticated, artistic skating expression",
        "chorus": "verse",
        "intensity": "medium"
    },
    "3Flip": {
        "prompt": "Dynamic orchestral piece with sudden dramatic accent, building anticipation then release, athletic and precise, competitive skating",
        "chorus": "chorus",
        "intensity": "high"
    },
    "3Lutz": {
        "prompt": "Powerful symphonic music with strong rhythmic drive, bold brass and percussion, confident and commanding, technical skating excellence",
        "chorus": "chorus",
        "intensity": "high"
    },
    "3Lutz_3Toeloop": {
        "prompt": "Explosive orchestral combination with dual climactic moments, relentless energy, virtuosic and breathtaking, elite figure skating combination jump",
        "chorus": "chorus",
        "intensity": "maximum"
    },
    "FlyCamelSpin4": {
        "prompt": "Ethereal flowing music with spinning melodic motifs, hypnotic circular patterns, dreamy and mesmerizing, graceful figure skating spin",
        "chorus": "verse",
        "intensity": "medium"
    },
    "ChComboSpin4": {
        "prompt": "Intricate classical music with layered spinning phrases, accelerating tempo, elegant complexity, artistic combination spin",
        "chorus": "verse",
        "intensity": "medium"
    },
    "ChoreoSequence1": {
        "prompt": "Expressive emotional music with lyrical melody, artistic interpretation, storytelling through sound, interpretive skating choreography",
        "chorus": "verse",
        "intensity": "low"
    },
    "StepSequence3": {
        "prompt": "Rhythmic intricate music with quick footwork patterns, playful yet precise, energetic dance-like quality, technical step sequence",
        "chorus": "verse",
        "intensity": "medium-high"
    },
    # Default for unknown moves
    "default": {
        "prompt": "Elegant classical figure skating music with graceful strings and piano, refined and artistic performance",
        "chorus": "verse",
        "intensity": "medium"
    }
}


# =============================================================================
# VIDEO ANALYSIS - SKELETON EXTRACTION
# =============================================================================

class SkeletonExtractor:
    """Extract skeleton poses from video using OpenPose"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / "skeletons"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_with_openpose(self, video_path: str) -> str:
        """Extract skeletons using OpenPose (if available)"""
        json_output_dir = self.output_dir / "json"
        json_output_dir.mkdir(exist_ok=True)
        
        openpose_bin = Path(self.config.openpose_dir) / "build/examples/openpose/openpose.bin"
        
        if not openpose_bin.exists():
            raise FileNotFoundError(f"OpenPose not found at {openpose_bin}")
        
        cmd = [
            str(openpose_bin),
            "--video", video_path,
            "--write_json", str(json_output_dir),
            "--display", "0",
            "--render_pose", "0",
            "--model_pose", "BODY_25"
        ]
        
        print(f"Running OpenPose: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        return str(json_output_dir)
    
    def extract_with_mmpose(self, video_path: str) -> str:
        """Extract skeletons using MMPose (alternative)"""
        # Placeholder for MMPose extraction
        # This would use MMPose's top-down or bottom-up pose estimation
        raise NotImplementedError("MMPose extraction not yet implemented")
    
    def json_to_npy(self, json_dir: str) -> str:
        """Convert OpenPose JSON outputs to numpy array format (C, T, V, M)"""
        json_files = sorted(Path(json_dir).glob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {json_dir}")
        
        num_frames = len(json_files)
        num_joints = 25  # OpenPose BODY_25
        num_channels = 3  # x, y, confidence
        num_persons = 1   # Single skater
        
        # Initialize array: (C, T, V, M)
        skeleton_data = np.zeros((num_channels, num_frames, num_joints, num_persons), dtype=np.float32)
        
        for frame_idx, json_file in enumerate(json_files):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if data.get("people") and len(data["people"]) > 0:
                keypoints = data["people"][0].get("pose_keypoints_2d", [])
                
                for joint_idx in range(num_joints):
                    base_idx = joint_idx * 3
                    if base_idx + 2 < len(keypoints):
                        skeleton_data[0, frame_idx, joint_idx, 0] = keypoints[base_idx]      # x
                        skeleton_data[1, frame_idx, joint_idx, 0] = keypoints[base_idx + 1]  # y
                        skeleton_data[2, frame_idx, joint_idx, 0] = keypoints[base_idx + 2]  # confidence
        
        output_path = str(self.output_dir / "skeleton_full.npy")
        np.save(output_path, skeleton_data)
        print(f" Skeleton data saved: {output_path} | Shape: {skeleton_data.shape}")
        
        return output_path
    
    def extract(self, video_path: str, method: str = "openpose") -> str:
        """Main extraction method"""
        print(f"\n{'='*60}")
        print("STEP 1: SKELETON EXTRACTION")
        print(f"{'='*60}")
        print(f"Video: {video_path}")
        print(f"Method: {method}")
        
        if method == "openpose":
            json_dir = self.extract_with_openpose(video_path)
        elif method == "mmpose":
            json_dir = self.extract_with_mmpose(video_path)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
        
        npy_path = self.json_to_npy(json_dir)
        return npy_path


# =============================================================================
# MOVE PREDICTION
# =============================================================================

class MovePredictorWrapper:
    """Wrapper for skating move prediction using GCN model"""
    
    LABELS = [
        "ChComboSpin4", "2Axel", "ChoreoSequence1", "3Loop", "StepSequence3",
        "3Flip", "FlyCamelSpin4", "3Axel", "3Lutz", "3Lutz_3Toeloop"
    ]
    
    # Penalties for over-predicted labels
    MUTED_PENALTIES = {
        "ChoreoSequence1": 12,
        "ChComboSpin4": 9
    }
    
    # Labels to filter out from final output
    IGNORE_LABELS = {"ChoreoSequence1", "ChComboSpin4"}
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.output_dir = Path(config.output_dir) / "predictions"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self):
        """Load the GCN model"""
        # Import model architecture
        try:
            from model.MDRGCN import Model
        except ImportError:
            print(" Model not found. Using mock predictions for demo.")
            return False
        
        self.model = Model(
            in_channels=3,
            num_class=len(self.LABELS),
            num_person=1,
            num_point=25,
            graph='graph.skating.Graph',
            graph_args={'layout': 'openpose25', 'strategy': 'spatial'}
        ).to(self.device)
        
        state_dict = torch.load(self.config.model_path, map_location=self.device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        
        print(f" Model loaded: {self.config.model_path}")
        return True
    
    def predict(self, skeleton_path: str) -> List[Dict]:
        """Run move prediction on skeleton data"""
        print(f"\n{'='*60}")
        print("STEP 2: MOVE PREDICTION")
        print(f"{'='*60}")
        
        skeletons = np.load(skeleton_path)
        C, T, V, M = skeletons.shape
        print(f"Skeleton shape: {skeletons.shape} | Total frames: {T} | Duration: {T/self.config.video_fps:.2f}s")
        
        results = []
        
        # Check if model is available
        model_available = self.load_model() if self.model is None else True
        
        if not model_available:
            # Generate mock predictions for demo
            results = self._generate_mock_predictions(T)
        else:
            results = self._run_inference(skeletons, T)
        
        # Post-process: merge consecutive same labels and filter
        merged_results = self._postprocess(results)
        
        # Save to JSON
        output_json = self.output_dir / "moves_prediction.json"
        with open(output_json, 'w') as f:
            json.dump(merged_results, f, indent=4)
        
        print(f" Predictions saved: {output_json}")
        print(f"   Detected {len(merged_results)} distinct moves")
        
        return merged_results
    
    def _run_inference(self, skeletons: np.ndarray, total_frames: int) -> List[Dict]:
        """Run actual model inference"""
        results = []
        
        with torch.no_grad():
            start = 0
            while start < total_frames:
                end = min(start + self.config.interval, total_frames)
                frame_len = end - start
                
                if frame_len < self.config.min_frames:
                    print(f"Skipping chunk {start}-{end} (too short: {frame_len} frames)")
                    break
                
                chunk = skeletons[:, start:end, :, :]
                
                # Pad if needed
                if chunk.shape[1] < self.config.model_window:
                    pad_width = self.config.model_window - chunk.shape[1]
                    chunk = np.pad(chunk, ((0, 0), (0, pad_width), (0, 0), (0, 0)), mode='constant')
                
                chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(self.device)
                
                _, output = self.model(chunk_tensor)
                
                # Apply penalties
                for lbl, penalty in self.MUTED_PENALTIES.items():
                    idx = self.LABELS.index(lbl)
                    output[:, idx] -= penalty
                
                probs = torch.softmax(output, dim=1)
                top1_prob, top1_idx = torch.topk(probs, k=1, dim=1)
                
                results.append({
                    "start_frame": start,
                    "end_frame": end,
                    "start_time": start / self.config.video_fps,
                    "end_time": end / self.config.video_fps,
                    "label": self.LABELS[top1_idx[0, 0].item()],
                    "confidence": top1_prob[0, 0].item()
                })
                
                start += self.config.interval
        
        return results
    
    def _generate_mock_predictions(self, total_frames: int) -> List[Dict]:
        """Generate mock predictions for demonstration"""
        print(" Using mock predictions (model not available)")
        
        # Simulate a typical skating program structure
        mock_moves = [
            ("3Lutz_3Toeloop", 0, 500),
            ("StepSequence3", 500, 1000),
            ("3Axel", 1000, 1500),
            ("FlyCamelSpin4", 1500, 2000),
            ("3Flip", 2000, 2500),
            ("3Loop", 2500, 3000),
            ("2Axel", 3000, 3500),
            ("StepSequence3", 3500, 4000),
        ]
        
        results = []
        for label, start, end in mock_moves:
            if start < total_frames:
                actual_end = min(end, total_frames)
                results.append({
                    "start_frame": start,
                    "end_frame": actual_end,
                    "start_time": start / self.config.video_fps,
                    "end_time": actual_end / self.config.video_fps,
                    "label": label,
                    "confidence": 0.85 + np.random.random() * 0.1
                })
        
        return results
    
    def _postprocess(self, results: List[Dict]) -> List[Dict]:
        """Merge consecutive same labels and filter ignored labels"""
        merged = []
        prev = None
        
        for r in results:
            label = r["label"]
            
            # Skip ignored labels
            if label in self.IGNORE_LABELS:
                continue
            
            if prev is None:
                prev = r.copy()
                continue
            
            # Merge if same label
            if label == prev["label"]:
                prev["end_time"] = r["end_time"]
                prev["end_frame"] = r["end_frame"]
            else:
                merged.append(self._format_result(prev))
                prev = r.copy()
        
        if prev is not None:
            merged.append(self._format_result(prev))
        
        return merged
    
    def _format_result(self, r: Dict) -> Dict:
        """Format result for output"""
        return {
            "move": r["label"],
            "start_time": round(r["start_time"], 2),
            "end_time": round(r["end_time"], 2),
            "duration": round(r["end_time"] - r["start_time"], 2),
            "confidence": round(r.get("confidence", 0.9), 4)
        }


# =============================================================================
# MUSIC GENERATION
# =============================================================================

class MusicGenerator:
    """Generate music for skating moves using InspireMusic"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / "music_segments"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
    
    def load_model(self):
        """Load InspireMusic model"""
        try:
            # Try to import InspireMusic
            sys.path.insert(0, str(Path(__file__).parent / "inspiremusic"))
            from inspiremusic.cli.inference import InspireMusicModel
            
            self.model = InspireMusicModel(
                model_name=self.config.inspiremusic_model,
                model_dir=self.config.inspiremusic_dir,
                min_generate_audio_seconds=self.config.min_music_duration,
                max_generate_audio_seconds=self.config.max_music_duration,
                output_sample_rate=self.config.output_sample_rate,
                gpu=self.config.gpu_id,
                result_dir=str(self.output_dir)
            )
            print(f" InspireMusic model loaded")
            return True
        except Exception as e:
            print(f" Could not load InspireMusic: {e}")
            return False
    
    def get_prompt_for_move(self, move: str) -> Dict:
        """Get music generation prompt for a skating move"""
        return MOVE_PROMPTS.get(move, MOVE_PROMPTS["default"])
    
    def generate_for_move(self, move: str, duration: float, segment_idx: int) -> Optional[str]:
        """Generate music for a single move"""
        prompt_data = self.get_prompt_for_move(move)
        
        # Clamp duration to valid range
        duration = max(self.config.min_music_duration, min(duration, self.config.max_music_duration))
        
        output_filename = f"segment_{segment_idx:03d}_{move}"
        
        print(f"   Generating: {move} ({duration:.1f}s)")
        print(f"   Prompt: {prompt_data['prompt'][:80]}...")
        
        if self.model is not None:
            try:
                output_path = self.model.inference(
                    task='text-to-music',
                    text=prompt_data['prompt'],
                    chorus=prompt_data['chorus'],
                    time_start=0.0,
                    time_end=duration,
                    output_fn=output_filename,
                    output_format="wav",
                    fade_out_mode=True
                )
                return output_path
            except Exception as e:
                print(f"    Generation failed: {e}")
                return None
        else:
            # Create placeholder for demo
            return self._create_placeholder(output_filename, duration)
    
    def _create_placeholder(self, filename: str, duration: float) -> str:
        """Create a silent placeholder audio file for demo"""
        output_path = self.output_dir / f"{filename}.wav"
        
        # Create silent audio segment
        silence = AudioSegment.silent(duration=int(duration * 1000))
        silence.export(str(output_path), format="wav")
        
        print(f"   Created placeholder: {output_path}")
        return str(output_path)
    
    def generate_all(self, moves: List[Dict]) -> List[Dict]:
        """Generate music for all detected moves"""
        print(f"\n{'='*60}")
        print("STEP 3: MUSIC GENERATION")
        print(f"{'='*60}")
        
        model_loaded = self.load_model()
        if not model_loaded:
            print(" Running in demo mode (generating placeholders)")
        
        generated_segments = []
        
        for idx, move_data in enumerate(moves):
            move = move_data["move"]
            duration = move_data["duration"]
            
            print(f"\n[{idx+1}/{len(moves)}] Processing: {move}")
            
            audio_path = self.generate_for_move(move, duration, idx)
            
            if audio_path:
                generated_segments.append({
                    "segment_idx": idx,
                    "move": move,
                    "start_time": move_data["start_time"],
                    "end_time": move_data["end_time"],
                    "duration": duration,
                    "audio_path": audio_path,
                    "prompt": self.get_prompt_for_move(move)["prompt"]
                })
        
        # Save generation log
        log_path = self.output_dir / "generation_log.json"
        with open(log_path, 'w') as f:
            json.dump(generated_segments, f, indent=4)
        
        print(f"\n Generated {len(generated_segments)} music segments")
        
        return generated_segments


# =============================================================================
# AUDIO CONCATENATION
# =============================================================================

class AudioConcatenator:
    """Concatenate music segments into final performance track"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / "final"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def concatenate(self, segments: List[Dict], output_filename: str = "performance_music") -> str:
        """Concatenate all segments with crossfade"""
        print(f"\n{'='*60}")
        print("STEP 4: AUDIO CONCATENATION")
        print(f"{'='*60}")
        
        if not segments:
            raise ValueError("No segments to concatenate")
        
        # Sort segments by start time
        segments = sorted(segments, key=lambda x: x["start_time"])
        
        # Load and concatenate
        combined = None
        
        for idx, seg in enumerate(segments):
            audio_path = seg["audio_path"]
            
            if not os.path.exists(audio_path):
                print(f" Skipping missing file: {audio_path}")
                continue
            
            print(f"   Adding segment {idx+1}: {seg['move']} ({seg['duration']:.1f}s)")
            
            segment_audio = AudioSegment.from_file(audio_path)
            
            if combined is None:
                combined = segment_audio
            else:
                # Apply crossfade for smooth transitions
                combined = combined.append(segment_audio, crossfade=self.config.crossfade_duration)
        
        if combined is None:
            raise ValueError("No valid audio segments to concatenate")
        
        # Normalize audio levels
        combined = combined.normalize()
        
        # Export final track
        output_path = self.output_dir / f"{output_filename}.wav"
        combined.export(str(output_path), format="wav")
        
        # Also export MP3 version
        mp3_path = self.output_dir / f"{output_filename}.mp3"
        combined.export(str(mp3_path), format="mp3", bitrate="320k")
        
        final_duration = len(combined) / 1000  # Convert ms to seconds
        
        print(f"\n Final track exported:")
        print(f"   WAV: {output_path}")
        print(f"   MP3: {mp3_path}")
        print(f"   Duration: {final_duration:.2f}s")
        
        return str(output_path)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class M2GPipeline:
    """Main M2G Pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.skeleton_extractor = SkeletonExtractor(config)
        self.move_predictor = MovePredictorWrapper(config)
        self.music_generator = MusicGenerator(config)
        self.audio_concatenator = AudioConcatenator(config)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def run(self, video_path: str = None, skeleton_path: str = None, moves_json: str = None) -> Dict:
        """
        Run the full pipeline
        
        Args:
            video_path: Path to input video (starts from step 1)
            skeleton_path: Path to skeleton .npy file (starts from step 2)
            moves_json: Path to moves JSON file (starts from step 3)
        
        Returns:
            Dictionary with all output paths and metadata
        """
        print("\n" + "="*60)
        print("M2G PIPELINE - Motion to Music Generation")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {
            "config": {
                "video_path": video_path,
                "skeleton_path": skeleton_path,
                "moves_json": moves_json,
                "output_dir": self.config.output_dir
            },
            "outputs": {}
        }
        
        # Step 1: Skeleton extraction (if starting from video)
        if video_path and not skeleton_path:
            try:
                skeleton_path = self.skeleton_extractor.extract(video_path)
                results["outputs"]["skeleton_path"] = skeleton_path
            except Exception as e:
                print(f" Skeleton extraction failed: {e}")
                print("   Please provide a pre-extracted skeleton file.")
                return results
        
        # Step 2: Move prediction (if starting from skeleton)
        if skeleton_path and not moves_json:
            moves = self.move_predictor.predict(skeleton_path)
            moves_json_path = self.config.output_dir + "/predictions/moves_prediction.json"
            results["outputs"]["moves_json"] = moves_json_path
        elif moves_json:
            with open(moves_json, 'r') as f:
                moves = json.load(f)
            print(f"\n Loaded {len(moves)} moves from: {moves_json}")
        else:
            raise ValueError("Must provide either video_path, skeleton_path, or moves_json")
        
        # Step 3: Music generation
        generated_segments = self.music_generator.generate_all(moves)
        results["outputs"]["music_segments"] = [s["audio_path"] for s in generated_segments]
        
        # Step 4: Audio concatenation
        if generated_segments:
            final_audio = self.audio_concatenator.concatenate(
                generated_segments,
                output_filename="performance_music"
            )
            results["outputs"]["final_audio"] = final_audio
        
        # Save pipeline results
        results_path = Path(self.config.output_dir) / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {results_path}")
        
        return results


# =============================================================================
# CLI INTERFACE
# =============================================================================

def get_user_input() -> Dict:
    """Interactive user input for pipeline configuration"""
    print("\n" + "="*60)
    print("M2G PIPELINE - Interactive Setup")
    print("="*60)
    
    print("\nSelect input type:")
    print("  1. Video file (full pipeline)")
    print("  2. Skeleton .npy file (skip extraction)")
    print("  3. Moves JSON file (music generation only)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    inputs = {
        "video_path": None,
        "skeleton_path": None,
        "moves_json": None
    }
    
    if choice == "1":
        inputs["video_path"] = input("Enter video file path: ").strip()
    elif choice == "2":
        inputs["skeleton_path"] = input("Enter skeleton .npy file path: ").strip()
    elif choice == "3":
        inputs["moves_json"] = input("Enter moves JSON file path: ").strip()
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    inputs["output_dir"] = input("Enter output directory (default: ./m2g_output): ").strip()
    if not inputs["output_dir"]:
        inputs["output_dir"] = "./m2g_output"
    
    return inputs


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="M2G Pipeline - Motion to Music Generation for Figure Skating",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline from video
  python m2g_pipeline.py --video skating_performance.mp4
  
  # Start from skeleton data
  python m2g_pipeline.py --skeleton skeleton_full.npy
  
  # Music generation only
  python m2g_pipeline.py --moves moves_prediction.json
  
  # Interactive mode
  python m2g_pipeline.py --interactive
        """
    )
    
    parser.add_argument('--video', '-v', type=str, help='Input video file path')
    parser.add_argument('--skeleton', '-s', type=str, help='Skeleton .npy file path')
    parser.add_argument('--moves', '-m', type=str, help='Moves JSON file path')
    parser.add_argument('--output', '-o', type=str, default='./m2g_output', help='Output directory')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (default: 0)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--model-path', type=str, default='./model/runs-215-21930.pt', help='GCN model path')
    parser.add_argument('--inspiremusic-dir', type=str, help='InspireMusic model directory')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Interactive mode
    if args.interactive or (not args.video and not args.skeleton and not args.moves):
        inputs = get_user_input()
        args.video = inputs.get("video_path")
        args.skeleton = inputs.get("skeleton_path")
        args.moves = inputs.get("moves_json")
        args.output = inputs.get("output_dir", args.output)
    
    # Create configuration
    config = PipelineConfig(
        output_dir=args.output,
        gpu_id=args.gpu,
        model_path=args.model_path
    )
    
    if args.inspiremusic_dir:
        config.inspiremusic_dir = args.inspiremusic_dir
    
    # Initialize and run pipeline
    pipeline = M2GPipeline(config)
    
    try:
        results = pipeline.run(
            video_path=args.video,
            skeleton_path=args.skeleton,
            moves_json=args.moves
        )
        
        print("\n Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":

    sys.exit(main())
