import argparse
import os
import time
import csv
import librosa
import sys
import io
from diarization import main  # from your updated diarization.py

def batch_process(input_folder, output_folder, num_files=None):
    os.makedirs(output_folder, exist_ok=True)

    audio_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.wav')])
    if num_files:
        audio_files = audio_files[:num_files]

    metrics_file = os.path.join(output_folder, "metrics.csv")
    with open(metrics_file, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'time_taken_s', 'num_segments', 'audio_duration_s', 'avg_segment_duration_s']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_start = time.time()

        for filename in audio_files:
            audio_path = os.path.join(input_folder, filename)
            print(f"\nProcessing {filename}...")

            start_time = time.time()

            # Capture diarization output
            old_stdout = sys.stdout
            sys.stdout = mystdout = io.StringIO()

            try:
                output_txt = os.path.join(output_folder, filename.replace('.wav', '.txt'))
                main(audio_path, output_txt)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
            finally:
                sys.stdout = old_stdout

            diarization_text = mystdout.getvalue()

            # Count segments and durations
            num_segments, total_segment_duration = parse_segments(diarization_text)

            try:
                audio_duration = librosa.get_duration(filename=audio_path)
            except Exception as e:
                print(f"Could not read duration for {filename}: {e}")
                audio_duration = 0.0

            elapsed = time.time() - start_time
            avg_segment_duration = total_segment_duration / num_segments if num_segments > 0 else 0

            print(f"Finished {filename} in {elapsed:.2f}s | Segments: {num_segments} | Duration: {audio_duration:.2f}s")

            writer.writerow({
                'filename': filename,
                'time_taken_s': f"{elapsed:.2f}",
                'num_segments': num_segments,
                'audio_duration_s': f"{audio_duration:.2f}",
                'avg_segment_duration_s': f"{avg_segment_duration:.2f}"
            })

        total_elapsed = time.time() - total_start
        print(f"\nTotal processing time: {total_elapsed:.2f}s for {len(audio_files)} files")

def parse_segments(text):
    num_segments = 0
    total_duration = 0.0
    for line in text.splitlines():
        if line.startswith("Speaker"):
            try:
                parts = line.split(":")[1].strip().split(" - ")
                start = float(parts[0].replace("s", ""))
                end = float(parts[1].replace("s", ""))
                total_duration += (end - start)
                num_segments += 1
            except Exception:
                continue
    return num_segments, total_duration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch speaker diarization using pyannote.audio")
    parser.add_argument("input_folder", type=str, help="Folder containing WAV audio files")
    parser.add_argument("output_folder", type=str, help="Folder to save diarization outputs and metrics")
    parser.add_argument("--num_files", type=int, default=None, help="Optional limit on number of files to process")
    args = parser.parse_args()

    batch_process(args.input_folder, args.output_folder, args.num_files)
