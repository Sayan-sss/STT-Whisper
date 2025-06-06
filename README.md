# 🎙️ STT-Whisper: Speech-to-Text using Hugging Face Whisper

This project uses OpenAI’s Whisper model via the Hugging Face `transformers` library to convert speech (audio files) into text. It's a lightweight, terminal-based tool that allows easy transcription with no UI overhead.

---

## 📁 Project Structure

```
stt-whisper/
├── audio_samples/              # Sample audio files for testing
│   ├── sample-0.mp3
│   └── sample-1.mp3
├── whisper_transcriber.py      # Main transcription script
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Requirements

- Python 3.9+
- `ffmpeg` installed system-wide
- Packages in `requirements.txt`

---

## 🧑‍💻 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stt-whisper.git
cd stt-whisper
```

### 2. Set Up a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg (Required for Audio Processing)

- **Windows**:
  ```powershell
  choco install ffmpeg
  ```
- **macOS (Homebrew)**:
  ```bash
  brew install ffmpeg
  ```
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt install ffmpeg
  ```

> ✅ Verify installation with:

```bash
ffmpeg -version
```

---

## 🚀 Usage

### Run the Transcriber

```bash
python whisper_transcriber.py audio_samples/sample-0.mp3
```

### Use a Different Whisper Model (Optional)

You can specify a different Hugging Face Whisper model:

```bash
python whisper_transcriber.py audio_samples/sample-0.mp3 --model openai/whisper-small
```

---

## 📝 Output

The script prints the transcribed text directly in the terminal:

```bash
📝 Transcription:

Hello, this is a sample test audio for the whisper model.
```

---

## 📦 Available Models on Hugging Face

You can try:

- `openai/whisper-tiny`
- `openai/whisper-base`
- `openai/whisper-small`
- `openai/whisper-medium`
- `openai/whisper-large`

Model size affects performance and accuracy.

---

## 📚 References

- 🔗 [Whisper on Hugging Face](https://huggingface.co/openai/whisper)
- 🔗 [Transformers Documentation](https://huggingface.co/docs/transformers)
- 🔗 [FFmpeg Installation Guide](https://ffmpeg.org/download.html)

---

## 👨‍💻 Author

**Sayan Mondal**  
[GitHub](https://github.com/Sayan-sss) • [LinkedIn](https://www.linkedin.com/in/sayan-mondal-10a734221/) • [Portfolio](https://sayan-portfolio-olive.vercel.app/)

---

## 📜 License

This project is licensed under the MIT License.
