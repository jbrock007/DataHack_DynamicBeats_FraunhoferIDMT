"""
Music Generator & Concatenator for M2G Pipeline
================================================
Standalone script that takes moves JSON and generates + concatenates music.

Usage:
    python generate_music.py --moves moves_prediction.json --output ./output
    python generate_music.py --interactive
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import subprocess

# Install required packages
def install_requirements():
    packages = ["pydub"]
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--break-system-packages", "--quiet"])

install_requirements()
from pydub import AudioSegment


# =============================================================================
# MOVE TO PROMPT MAPPING
# =============================================================================

MOVE_PROMPTS = {
    "2Axel": {
        "prompt": "Dramatic orchestral music with building tension, powerful brass crescendo, elegant and athletic energy, figure skating performance",
        "chorus": "chorus",
        "intensity": "high",
        "tempo": "moderate-fast"
    },
    "3Axel": {
        "prompt": "Epic cinematic orchestral score with soaring strings, triumphant brass fanfare, maximum intensity and grandeur, championship figure skating moment",
        "chorus": "chorus", 
        "intensity": "maximum",
        "tempo": "fast"
    },
    "3Loop": {
        "prompt": "Graceful flowing classical music with swirling strings, elegant waltz-like melody, refined and sophisticated, artistic skating expression",
        "chorus": "verse",
        "intensity": "medium",
        "tempo": "moderate"
    },
    "3Flip": {
        "prompt": "Dynamic orchestral piece with sudden dramatic accent, building anticipation then release, athletic and precise, competitive skating",
        "chorus": "chorus",
        "intensity": "high",
        "tempo": "moderate-fast"
    },
    "3Lutz": {
        "prompt": "Powerful symphonic music with strong rhythmic drive, bold brass and percussion, confident and commanding, technical skating excellence",
        "chorus": "chorus",
        "intensity": "high",
        "tempo": "fast"
    },
    "3Lutz_3Toeloop": {
        "prompt": "Explosive orchestral combination with dual climactic moments, relentless energy, virtuosic and breathtaking, elite figure skating combination jump",
        "chorus": "chorus",
        "intensity": "maximum",
        "tempo": "very-fast"
    },
    "FlyCamelSpin4": {
        "prompt": "Ethereal flowing music with spinning melodic motifs, hypnotic circular patterns, dreamy and mesmerizing, graceful figure skating spin",
        "chorus": "verse",
        "intensity": "medium",
        "tempo": "moderate"
    },
    "ChComboSpin4": {
        "prompt": "Intricate classical music with layered spinning phrases, accelerating tempo, elegant complexity, artistic combination spin",
        "chorus": "verse",
        "intensity": "medium",
        "tempo": "accelerating"
    },
    "ChoreoSequence1": {
        "prompt": "Expressive emotional music with lyrical melody, artistic interpretation, storytelling through sound, interpretive skating choreography",
        "chorus": "verse",
        "intensity": "low-medium",
        "tempo": "slow-moderate"
    },
    "StepSequence3": {
        "prompt": "Rhythmic intricate music with quick footwork patterns, playful yet precise, energetic dance-like quality, technical step sequence",
        "chorus": "verse",
        "intensity": "medium-high",
        "tempo": "fast"
    },
    "default": {
        "prompt": "Elegant classical figure skating music with graceful strings and piano, refined and artistic performance",
        "chorus": "verse",
        "intensity": "medium",
        "tempo": "moderate"
    }
}


# =============================================================================
# INSPIRE MUSIC WRAPPER
# =============================================================================

class InspireMusicWrapper:
    """Wrapper for InspireMusic model"""
    
    def __init__(self, 
                 model_name: str = "InspireMusic-1.5B-Long",
                 model_dir: str = None,
                 output_dir: str = "./music_output",
                 gpu: int = 0,
                 min_duration: float = 5.0,
                 max_duration: float = 30.0):
        
        self.model_name = model_name
        self.model_dir = model_dir or f"./pretrained_models/{model_name}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu = gpu
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.model = None
        self.available = False
    
    def load(self) -> bool:
        """Load InspireMusic model"""
        try:
            # Add InspireMusic to path
            inspire_path = Path(__file__).parent / "inspiremusic"
            if inspire_path.exists():
                sys.path.insert(0, str(inspire_path))
            
            from inspiremusic.cli.inference import InspireMusicModel
            
            self.model = InspireMusicModel(
                model_name=self.model_name,
                model_dir=self.model_dir,
                min_generate_audio_seconds=self.min_duration,
                max_generate_audio_seconds=self.max_duration,
                output_sample_rate=48000,
                gpu=self.gpu,
                result_dir=str(self.output_dir)
            )
            self.available = True
            print(f" InspireMusic loaded: {self.model_name}")
            return True
            
        except ImportError as e:
            print(f" InspireMusic not available: {e}")
            print("   Will create placeholder audio files for demo.")
            self.available = False
            return False
        except Exception as e:
            print(f" Failed to load InspireMusic: {e}")
            self.available = False
            return False
    
    def generate(self, 
                 prompt: str, 
                 duration: float, 
                 output_name: str,
                 chorus: str = "verse") -> Optional[str]:
        """Generate music from prompt"""
        
        # Clamp duration
        duration = max(self.min_duration, min(duration, self.max_duration))
        
        if self.available and self.model:
            try:
                output_path = self.model.inference(
                    task='text-to-music',
                    text=prompt,
                    chorus=chorus,
                    time_start=0.0,
                    time_end=duration,
                    output_fn=output_name,
                    output_format="wav",
                    fade_out_mode=True,
                    fade_out_duration=1.0
                )
                return output_path
            except Exception as e:
                print(f"   ❌ Generation failed: {e}")
                return self._create_placeholder(output_name, duration)
        else:
            return self._create_placeholder(output_name, duration)
    
    def _create_placeholder(self, name: str, duration: float) -> str:
        """Create silent placeholder audio"""
        output_path = self.output_dir / f"{name}.wav"
        silence = AudioSegment.silent(duration=int(duration * 1000))
        silence.export(str(output_path), format="wav")
        return str(output_path)


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class MusicPipelineGenerator:
    """Generate and concatenate music for skating performance"""
    
    def __init__(self, 
                 output_dir: str = "./m2g_output",
                 gpu: int = 0,
                 crossfade_ms: int = 1000,
                 min_duration: float = 5.0,
                 max_duration: float = 30.0):
        
        self.output_dir = Path(output_dir)
        self.segments_dir = self.output_dir / "segments"
        self.final_dir = self.output_dir / "final"
        
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        
        self.crossfade_ms = crossfade_ms
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        self.inspire_music = InspireMusicWrapper(
            output_dir=str(self.segments_dir),
            gpu=gpu,
            min_duration=min_duration,
            max_duration=max_duration
        )
    
    def load_moves(self, moves_path: str) -> List[Dict]:
        """Load moves from JSON file"""
        with open(moves_path, 'r') as f:
            moves = json.load(f)
        
        print(f" Loaded {len(moves)} moves from: {moves_path}")
        return moves
    
    def get_prompt(self, move: str) -> Dict:
        """Get prompt configuration for a move"""
        return MOVE_PROMPTS.get(move, MOVE_PROMPTS["default"])
    
    def generate_segments(self, moves: List[Dict]) -> List[Dict]:
        """Generate music for all moves"""
        print(f"\n{'='*60}")
        print("GENERATING MUSIC SEGMENTS")
        print(f"{'='*60}\n")
        
        # Load model
        self.inspire_music.load()
        
        segments = []
        
        for idx, move_data in enumerate(moves):
            move = move_data["move"]
            duration = move_data.get("duration", 
                                     move_data["end_time"] - move_data["start_time"])
            
            prompt_config = self.get_prompt(move)
            
            print(f"[{idx+1}/{len(moves)}] {move} ({duration:.1f}s)")
            print(f"    Prompt: {prompt_config['prompt'][:60]}...")
            
            output_name = f"seg_{idx:03d}_{move}"
            
            audio_path = self.inspire_music.generate(
                prompt=prompt_config["prompt"],
                duration=duration,
                output_name=output_name,
                chorus=prompt_config["chorus"]
            )
            
            if audio_path:
                segments.append({
                    "index": idx,
                    "move": move,
                    "start_time": move_data["start_time"],
                    "end_time": move_data["end_time"],
                    "duration": duration,
                    "audio_path": audio_path,
                    "prompt": prompt_config["prompt"]
                })
                print(f"     Saved: {audio_path}")
            else:
                print(f"     Failed to generate")
        
        # Save segments log
        log_path = self.segments_dir / "segments_log.json"
        with open(log_path, 'w') as f:
            json.dump(segments, f, indent=4)
        
        print(f"\n✅ Generated {len(segments)} segments")
        return segments
    
    def concatenate(self, 
                    segments: List[Dict], 
                    output_name: str = "performance_music") -> Dict:
        """Concatenate all segments into final track"""
        print(f"\n{'='*60}")
        print("CONCATENATING AUDIO")
        print(f"{'='*60}\n")
        
        if not segments:
            raise ValueError("No segments to concatenate")
        
        # Sort by start time
        segments = sorted(segments, key=lambda x: x["start_time"])
        
        combined = None
        
        for seg in segments:
            audio_path = seg["audio_path"]
            
            if not os.path.exists(audio_path):
                print(f" Missing: {audio_path}")
                continue
            
            print(f"   Adding: {seg['move']} ({seg['duration']:.1f}s)")
            
            audio = AudioSegment.from_file(audio_path)
            
            if combined is None:
                combined = audio
            else:
                combined = combined.append(audio, crossfade=self.crossfade_ms)
        
        if combined is None:
            raise ValueError("No valid audio to concatenate")
        
        # Normalize
        combined = combined.normalize()
        
        # Export WAV
        wav_path = self.final_dir / f"{output_name}.wav"
        combined.export(str(wav_path), format="wav")
        
        # Export MP3
        mp3_path = self.final_dir / f"{output_name}.mp3"
        combined.export(str(mp3_path), format="mp3", bitrate="320k")
        
        final_duration = len(combined) / 1000
        
        print(f"\n Final track exported:")
        print(f"   WAV: {wav_path}")
        print(f"   MP3: {mp3_path}")
        print(f"   Duration: {final_duration:.2f}s")
        
        return {
            "wav_path": str(wav_path),
            "mp3_path": str(mp3_path),
            "duration": final_duration,
            "num_segments": len(segments)
        }
    
    def run(self, moves_path: str, output_name: str = "performance_music") -> Dict:
        """Run full generation and concatenation pipeline"""
        print("\n" + "="*60)
        print("M2G MUSIC GENERATION PIPELINE")
        print("="*60)
        
        # Load moves
        moves = self.load_moves(moves_path)
        
        # Display moves summary
        print("\nDetected Moves:")
        for m in moves:
            print(f"   {m['start_time']:6.2f}s - {m['end_time']:6.2f}s : {m['move']}")
        
        # Generate segments
        segments = self.generate_segments(moves)
        
        # Concatenate
        result = self.concatenate(segments, output_name)
        
        # Save final results
        results = {
            "input_moves": moves_path,
            "output": result,
            "segments": segments
        }
        
        results_path = self.final_dir / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Results: {results_path}")
        
        return results


# =============================================================================
# CLI
# =============================================================================

def interactive_mode():
    """Interactive mode for user input"""
    print("\n" + "="*60)
    print("M2G Music Generator - Interactive Mode")
    print("="*60)
    
    moves_path = input("\nEnter moves JSON file path: ").strip()
    
    if not os.path.exists(moves_path):
        print(f"❌ File not found: {moves_path}")
        sys.exit(1)
    
    output_dir = input("Output directory (default: ./m2g_output): ").strip()
    if not output_dir:
        output_dir = "./m2g_output"
    
    output_name = input("Output filename (default: performance_music): ").strip()
    if not output_name:
        output_name = "performance_music"
    
    return moves_path, output_dir, output_name


def main():
    parser = argparse.ArgumentParser(
        description="Generate and concatenate music for figure skating performance"
    )
    
    parser.add_argument('--moves', '-m', type=str, help='Moves JSON file path')
    parser.add_argument('--output', '-o', type=str, default='./m2g_output', help='Output directory')
    parser.add_argument('--name', '-n', type=str, default='performance_music', help='Output filename')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID')
    parser.add_argument('--crossfade', type=int, default=1000, help='Crossfade duration in ms')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive or not args.moves:
        moves_path, output_dir, output_name = interactive_mode()
    else:
        moves_path = args.moves
        output_dir = args.output
        output_name = args.name
    
    # Run pipeline
    generator = MusicPipelineGenerator(
        output_dir=output_dir,
        gpu=args.gpu,
        crossfade_ms=args.crossfade
    )
    
    try:
        results = generator.run(moves_path, output_name)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())