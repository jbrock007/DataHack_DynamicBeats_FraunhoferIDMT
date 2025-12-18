import torch
import numpy as np
from model.MDRGCN import Model
import json

class SkatingMovePredictor:
    def __init__(self, model_path, skeleton_file, device=None):
        # --- CONFIG ---
        self.MODEL_PATH = model_path
        self.VIDEO_SKELETONS = skeleton_file
        self.DEVICE = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.LABELS = [
            "ChComboSpin4", "2Axel", "ChoreoSequence1", "3Loop", "StepSequence3",
            "3Flip", "FlyCamelSpin4", "3Axel", "3Lutz", "3Lutz_3Toeloop"
        ]

        # --- Independent penalties for specific labels ---
        # Larger = stronger penalty (lower confidence)
        self.MUTED_PENALTIES = {
            "ChoreoSequence1": 12,   # strong penalty
            "ChComboSpin4": 9        # moderate penalty
        }

        self.MODEL_WINDOW = 500
        self.INTERVAL = 500
        self.VIDEO_FPS = 30
        self.MIN_FRAMES = 300   # ignore last chunk shorter than this

        # --- LOAD SKELETON DATA ---
        self.skeletons = np.load(self.VIDEO_SKELETONS)  # (C, T, V, M)
        self.C, self.T, self.V, self.M = self.skeletons.shape

        # --- BUILD MODEL ---
        self.model = Model(
            in_channels=3,
            num_class=len(self.LABELS),
            num_person=1,
            num_point=25,
            graph='graph.skating.Graph',
            graph_args={'layout': 'openpose25', 'strategy': 'spatial'}
        ).to(self.DEVICE)

        # --- LOAD WEIGHTS ---
        state_dict = torch.load(self.MODEL_PATH, map_location=self.DEVICE)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def run_prediction(self):
        results = []

        with torch.no_grad():
            start = 0
            while start < self.T:
                end = min(start + self.INTERVAL, self.T)
                frame_len = end - start

                # ✅ Skip if chunk too short
                if frame_len < self.MIN_FRAMES:
                    print(f"Skipping chunk {start}-{end} (too short: {frame_len} frames)")
                    break

                chunk = self.skeletons[:, start:end, :, :]

                # Pad if too short for model window
                if chunk.shape[1] < self.MODEL_WINDOW:
                    pad_width = self.MODEL_WINDOW - chunk.shape[1]
                    chunk = np.pad(chunk, ((0, 0), (0, pad_width), (0, 0), (0, 0)),
                                   mode='constant', constant_values=0)

                chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(self.DEVICE)

                # --- Run model once ---
                _, output = self.model(chunk_tensor)

                # --- Apply independent penalties ---
                for lbl, penalty in self.MUTED_PENALTIES.items():
                    idx = self.LABELS.index(lbl)
                    output[:, idx] -= penalty

                # --- Convert logits to probabilities ---
                probs = torch.softmax(output, dim=1)

                # --- Get top prediction ---
                top1_prob, top1_idx = torch.topk(probs, k=1, dim=1)
                top1_idx_val = top1_idx[0, 0].item()
                top1_prob_val = top1_prob[0, 0].item()

                # --- Store prediction ---
                results.append({
                    "start_frame": start,
                    "end_frame": end,
                    "start_time": start / self.VIDEO_FPS,
                    "end_time": end / self.VIDEO_FPS,
                    "top_label": self.LABELS[top1_idx_val],
                    "top_confidence": top1_prob_val
                })

                # Increment by full INTERVAL (no overlap)
                start += self.INTERVAL

        return results

    def print_results(self, results):
        # --- PRINT RESULTS ---
        print("\n=== Final Prediction Results (Single Pass, Short Chunks Ignored) ===")
        for r in results:
            print(f"Frames {r['start_frame']}-{r['end_frame']} "
                  f"({r['start_time']:.2f}s-{r['end_time']:.2f}s): "
                  f"Top1: {r['top_label']} ({r['top_confidence']:.4f})")
            print("-" * 40)


    def save_json(self, results, output_file="../output/moves_prediction.json"):
        IGNORE_LABELS = {"ChoreoSequence1", "ChComboSpin4"}

        # --- POST-PROCESSING ---
        merged_results = []

        prev = None

        for r in results:
            label = r["top_label"]
            if label in IGNORE_LABELS:
                continue

            if prev is None:
                prev = r.copy()
                continue

            # If same label as previous, merge time ranges
            if label == prev["top_label"]:
                prev["start_time"] = min(prev["start_time"], r["start_time"])
                prev["end_time"] = max(prev["end_time"], r["end_time"])
            else:
                merged_results.append(prev)
                prev = r.copy()

        # Append the last one
        if prev is not None:
            merged_results.append(prev)

        # --- CONVERT TO JSON ---
        json_output = []
        for r in merged_results:
            json_output.append({
                "move": r["top_label"],
                "start_time": round(r["start_time"], 2),
                "end_time": round(r["end_time"], 2),
                "mean_time": int(round(r["start_time"], 2) + round(r["end_time"], 2)) / 2
            })

        with open(output_file, "w") as f:
            json.dump(json_output, f, indent=4)

        print(f"✅ JSON saved with merged moves: {json_output}")


# --- USAGE ---
if __name__ == "__main__":
    predictor = SkatingMovePredictor("../model/runs-215-21930.pt", "../data/skeleton_full.npy")
    results = predictor.run_prediction()
    predictor.print_results(results)
    predictor.save_json(results)
