
# YouTube Video Generator

A Python application to generate YouTube scripts, voiceovers, and videos using AI and Pixabay resources.

---

## 📦 Features

- Generate YouTube scripts using Google Generative AI
- Generate text-to-speech voiceovers
- Download images or videos automatically
- Create complete YouTube videos by combining audio and visuals
- Modern tkinter GUI interface
- Progress bar and real-time status updates

---

## 🔧 Setup Instructions

### 1. **Clone the repository**
First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. **Create and activate a virtual environment (optional but recommended)**

On **Windows**:
```bash
python -m venv venv
.env\Scriptsctivate
```

On **Linux/Mac**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. **Install the required Python packages**
Once the virtual environment is activated, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 4. **Install FFmpeg**
FFmpeg is required for video processing. Follow the instructions below to install it:

#### Windows:
- Download FFmpeg from the official website: [FFmpeg Download](https://ffmpeg.org/download.html).
- Extract the ZIP file and add the `bin/` directory to your system's PATH environment variable.

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install ffmpeg
```

#### macOS (using Homebrew):
```bash
brew install ffmpeg
```

### 5. **Configure your API keys**
- Copy `config.example.py` to `config.py`.
- Open `config.py` and add your actual API keys.

### 6. **Run the application**

Once everything is set up, you can run the application using:

```bash
python main.py
```

## 📜 License

This project is licensed under the [MIT License](LICENSE).
