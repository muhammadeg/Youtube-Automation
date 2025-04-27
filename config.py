# Put all shared configuration and constants here
import os

CHUNK_SIZE = 1024

# Folder paths
DOCX_FOLDER = "Docx"
AUDIO_FOLDER = "Audio"
VISUAL_FOLDER = "visual"
VIDEO_FOLDER = "Videos"
CROPPED_FOLDER = "cropped"
CAPTIONS_FOLDER = "Captions"
# API Keys
GENAI_API_KEY = 'INSERT_API_KEY_HERE'
ELEVENLABS_API_KEY = 'INSERT_API_KEY_HERE'
PEXELS_API_KEY = 'INSERT_API_KEY_HERE'
PIXABAY_API_KEY = '48372141-INSERT_API_KEY_HERE'
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/YOUR_ID"

# Ensure the folder exists
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(DOCX_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(VISUAL_FOLDER, exist_ok=True)
#
# DEFAULT_PROMPT = (
#     "Write a unique 10-second script for a YouTube video that delivers fresh daily financial tips and impactful life lessons. "
#     "Start the script with an engaging opening sentence that captures the essence of the video without explicitly repeating or restating the title. "
#     "The title should inspire the tone and focus of the script but must not be repeated verbatim in the narration. "
#     "Each script should vary in tone, perspective, and examples to ensure no repetition across multiple requests. "
#     "The body of the script should deliver actionable advice, inspiring insights, or surprising facts, written in a way that an AI voice-over can read seamlessly. "
#     "Avoid generic advice and provide specific, original, and thought-provoking content in a positive and uplifting tone. "
#     "The script must not include stage directions, music cues, labels like 'Title:', or any non-narrative elements. "
#     "Focus on motivating the audience to make smarter financial choices, embrace life improvements, and take small steps toward their dreams."
# )
# DEFAULT_PROMPT = (
#     "Write a unique 10-second script for a YouTube video featuring a lighthearted and entertaining story in a relatable, everyday setting. "
#     "Start the script with an engaging opening sentence that captures the essence of the story without explicitly repeating or restating the title. "
#     "The title should inspire the tone and focus of the script but must not be repeated verbatim in the narration. "
#     "Each script should vary in tone, perspective, and examples to ensure no repetition across multiple requests. "
#     "The body of the script should deliver witty dialogue, humorous misunderstandings, or quirky observations written in a way that an AI voice-over can read seamlessly. "
#     "Avoid generic jokes, overused punchlines, and include fresh, original, and thought-provoking content in an entertaining and positive tone. "
#     "The script must not include stage directions, music cues, labels like 'Title:', or any non-narrative elements. "
#     "Focus on creating a fun and memorable experience for the audience, leaving them with a laugh or a clever twist in just a few lines."
# )

DEFAULT_PROMPT = (
    "Write a unique 30-second script for a YouTube video featuring a mind-blowing and fascinating fun facts. "
    "Start the script with an engaging opening sentence that captures the intrigue of the fact without explicitly repeating or restating the title. "
    "The title should inspire the tone and focus of the script but must not be repeated verbatim in the narration. "
    "Each script should vary in tone, perspective, and examples to ensure no repetition across multiple requests. "
    "The body of the script should deliver the fun fact in a witty, unexpected, or quirky way, written seamlessly for an AI voice-over. "
    "Avoid generic phrases, overused clichés, and include fresh, original, and thought-provoking content in an entertaining and positive tone. "
    "The script must not include stage directions, music cues, labels like 'Title:', or any non-narrative elements. "
    "Focus on creating a fun and memorable experience for the audience, leaving them intrigued, amazed, or amused in just a few lines."
)



