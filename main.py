import os
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import font
import google.generativeai as genai
from utils import timestamp, generate_script, download_images,  download_and_crop_videos_from_pixabay, \
    combine_pixabay, create_combined_video
from config import GENAI_API_KEY, AUDIO_FOLDER, VIDEO_FOLDER, CAPTIONS_FOLDER

genai.configure(api_key=GENAI_API_KEY)

# GUI setup
root = tk.Tk()
root.title("YouTube Script and Voice Generator")
root.geometry("1920x900")  # Adjusted window size for better layout
root.config(bg="#2E2E2E")  # Set the background color for the root window

# Define fonts and styles
header_font = font.Font(family="Helvetica", size=16, weight="bold")
button_font = font.Font(family="Arial", size=12)
status_font = font.Font(family="Helvetica", size=12)

# Define dark theme colors
dark_bg = "#2E2E2E"
dark_fg = "#FFFFFF"
button_bg = "#4CAF50"
button_active_bg = "#45a049"
button_fg = "#FFFFFF"
button_hover_bg = "#388E3C"

# Status label to show progress
status_label = tk.Label(root, text="Welcome! Ready to Generate.", font=status_font, anchor="center", bg=dark_bg,
                        fg=dark_fg)
status_label.grid(row=0, column=0, columnspan=3, pady=20, sticky="nsew")

# Progress Bar for feedback on long tasks
progress = ttk.Progressbar(root, length=300, mode="determinate", maximum=100)
progress.grid(row=1, column=0, columnspan=3, pady=10, sticky="nsew")

# Function to update status and show progress
def update_status(message, progress_value=None):
    status_label.config(text=message)
    if progress_value is not None:
        progress['value'] = progress_value
        root.update_idletasks()


# Threaded execution wrapper
def threaded_execution(target):
    thread = threading.Thread(target=target)
    thread.start()

# Function to handle script generation
def handle_generate_script(voice_required):
    def task():
        update_status("Generating script...", progress_value=50)
        generate_script(voice_required=voice_required)
        update_status("Script generation complete.", progress_value=100)

    threaded_execution(task)


# Function to handle script + voice generation
def handle_generate_script_and_voice():
    def task():
        update_status("Generating script and voice...", progress_value=50)
        generate_script(voice_required=True)
        update_status("Script and voice generation complete.", progress_value=100)

    threaded_execution(task)


# Function to handle image download + video creation
def handle_generate_file_voice_images_and_video():
    def task():
        update_status("Generating script, voice, and downloading images...", progress_value=25)
        generate_script(voice_required=True)
        update_status("Script & Voice Generated.. Downloading Images", progress_value=50)
        download_images()
        update_status("Images downloaded. Creating video...", progress_value=50)

        create_combined_video(
            os.path.join(AUDIO_FOLDER, f"File_1_{timestamp}.mp3"),
            os.path.join(VIDEO_FOLDER, f"File_1_{timestamp}.mp4")
        )
        update_status("Process complete. Video generated.", progress_value=100)

    threaded_execution(task)


def handle_generate_file_voice_videos_and_video():
    def task():
        update_status("Generating script & voice...", progress_value=0)
        generate_script(voice_required=True)
        update_status("Script & Voice generated. Downloading videos...", progress_value=25)

        download_and_crop_videos_from_pixabay()
        update_status("Videos downloaded. Creating video...", progress_value=50)

        combine_pixabay(
            audio_file=os.path.join(AUDIO_FOLDER, f"File_1_{timestamp}.mp3"),
            output_file=os.path.join(VIDEO_FOLDER, f"File_1_{timestamp}.mp4"),
            visual_folder="cropped",
            current_timestamp=timestamp,
        )

        update_status("Video creation complete. Finalizing...", progress_value=75)

        update_status("Process complete. Video generated!", progress_value=100)

    threaded_execution(task)


# Helper function to style buttons with modern look
def modern_button(frame, text, command):
    button = tk.Button(
        frame, text=text, font=button_font, command=command, width=25, height=2,
        relief="flat", bg=button_bg, fg=button_fg, activebackground=button_active_bg, activeforeground=button_fg,
        bd=2, padx=15, pady=10, highlightthickness=0, anchor="center"
    )

    # Button hover effect (temporary)
    def on_enter(event):
        button.config(bg=button_hover_bg)

    def on_leave(event):
        button.config(bg=button_bg)

    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)

    return button


# Frame for main actions
frame1 = tk.Frame(root, padx=20, pady=20, bg=dark_bg, bd=2, relief="solid", highlightbackground="#AAAAAA",
                  highlightcolor="#AAAAAA")
frame1.grid(row=2, column=0, columnspan=3, pady=20, sticky="nsew")

# Ensure the frame can expand
frame1.grid_rowconfigure(0, weight=1)
frame1.grid_columnconfigure(0, weight=1)
frame1.grid_columnconfigure(1, weight=1)
frame1.grid_columnconfigure(2, weight=1)

# Buttons for main actions (centered)
generate_file_button = modern_button(frame1, "Generate Script", lambda: handle_generate_script(voice_required=False))
generate_file_button.grid(row=0, column=0, pady=10, padx=10, columnspan=3)

generate_file_and_voice_button = modern_button(frame1, "Generate Script & Voice", handle_generate_script_and_voice)
generate_file_and_voice_button.grid(row=1, column=0, pady=10, padx=10, columnspan=3)

generate_file_voice_images_and_video_button = modern_button(frame1, "Generate Video by Images",
                                                            handle_generate_file_voice_images_and_video)
generate_file_voice_images_and_video_button.grid(row=2, column=0, pady=10, padx=10, columnspan=3)

generate_file_voice_videos_and_video_button = modern_button(frame1, "Generate Full Video",
                                                            handle_generate_file_voice_videos_and_video)
generate_file_voice_videos_and_video_button.grid(row=3, column=0, pady=10, padx=10, columnspan=3)

# Frame for download actions (stacked below the first frame)
frame2 = tk.Frame(root, padx=20, pady=20, bg=dark_bg, bd=2, relief="solid", highlightbackground="#AAAAAA",
                  highlightcolor="#AAAAAA")
frame2.grid(row=3, column=0, columnspan=3, pady=20, sticky="nsew")

# Ensure the frame can expand
frame2.grid_rowconfigure(0, weight=1)
frame2.grid_columnconfigure(0, weight=1)
frame2.grid_columnconfigure(1, weight=1)
frame2.grid_columnconfigure(2, weight=1)

# Buttons for download actions (centered)
download_images_button = modern_button(frame2, "Download Images", lambda: download_images())
download_images_button.grid(row=0, column=0, pady=10, padx=10, columnspan=3)

download_videos_button = modern_button(frame2, "Download Videos", lambda: download_and_crop_videos_from_pixabay())
download_videos_button.grid(row=1, column=0, pady=10, padx=10, columnspan=3)

# Configure grid layout for the root window
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)

root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)

root.mainloop()
