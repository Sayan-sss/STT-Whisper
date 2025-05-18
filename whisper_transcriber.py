import argparse
from transformers import pipeline

def transcribe_audio(audio_path: str, model_name: str = "openai/whisper-base"):
    """Load Whisper from Hugging Face and transcribe audio."""
    print(f"\n🔄 Loading Hugging Face model: {model_name} ...")
    pipe = pipeline(task="automatic-speech-recognition", model=model_name)

    print(f"🎧 Transcribing audio: {audio_path} ...")
    result = pipe(audio_path)

    transcript = result["text"]
    print("\n📝 Transcription:\n")
    print(transcript)

    # Save to a text file
    with open("transcription.txt", "w", encoding="utf-8") as f:
        f.write(transcript)
    print("\n✅ Transcription saved to: transcription.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🎙️ Transcribe audio using Hugging Face Whisper")
    parser.add_argument("audio_path", help="Path to your input audio file (mp3/wav/m4a)")
    parser.add_argument(
        "--model",
        default="openai/whisper-base",
        help="Hugging Face model ID (default: openai/whisper-base)"
    )

    args = parser.parse_args()
    transcribe_audio(args.audio_path, args.model)
