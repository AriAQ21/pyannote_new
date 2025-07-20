import argparse
import os
from pyannote.audio import Pipeline

def main(audio_path, output_txt=None):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not found in environment. Please set it.")

    # Use the latest diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)

    # Run diarization
    diarization = pipeline(audio_path)

    # Format results
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        line = f"Speaker {speaker}: {turn.start:.2f}s - {turn.end:.2f}s"
        print(line)
        results.append(line)

    # Write to output
    if output_txt:
        with open(output_txt, "w") as f:
            f.write("\n".join(results))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pyannote.audio speaker diarization (v3.1)")
    parser.add_argument("audio", type=str, help="Path to input WAV audio file")
    parser.add_argument("--output", type=str, default=None, help="Path to output text file")
    args = parser.parse_args()
    main(args.audio, args.output)
