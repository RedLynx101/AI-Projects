from openai import OpenAI
import json
import os 
from tkinter import Tk, filedialog

# This script uses the OpenAI API to transcribe an audio file to text.

# Use tkinter to select an audio file to transcribe
try:
    def select_audio_file():
        # Create a Tkinter root window (hidden)
        root = Tk()
        root.withdraw()

        # Allow the user to select a file and limit file types to common audio/video formats
        audio_file_path = filedialog.askopenfilename(
            title="Select an audio file",
            filetypes=[("Audio Files", "*.mp3 *.mp4 *.wav *.mkv *.ogg *.flac")]
        )

        if audio_file_path:
            print(f"System: File selected: {audio_file_path}")
            return audio_file_path
        else:
            print("System: No file selected. Exiting...")
            exit()
except Exception as e:
    print(f"System: Error selecting audio file: {e}")
    # Exit the program if an error occurs
    exit()


# Main program following steps to transcribe the audio file using the OpenAI API
try:
    # Your OpenAI API key, reads from a JSON file 'openaikey'
    try:
        with open('openaikey.json') as f:
            data = json.load(f)
            OPENAI_API_KEY = data['openai_api_key']
            print("System: The key was loaded successfully.")
    except Exception as e:
        print(f"System: Error loading OpenAI API key: {e}")
        # Exit the program if the API key is not loaded
        exit()

    # Set your OpenAI API key
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("System: OpenAI API key set successfully.")
    except Exception as e:
        print(f"System: Error setting OpenAI API key: {e}")
        # Exit the program if the API key is not set
        exit()

    # Select an audio file using the file dialog
    try:
        audio_file_path = select_audio_file()  # Call the file selection function
    except Exception as e:
        print(f"System: Error selecting audio file: {e}")
        # Exit the program if the audio file path is not selected
        exit()

    # Open the file and send it for transcription
    try:
        with open(audio_file_path, "rb") as audio_file:
            print("System: Sending transcription request...")
            transcript = client.audio.transcriptions.create(file=audio_file, model="whisper-1", response_format="text")
            print("System: Transcription request sent successfully.")
    except Exception as e:
        print(f"System: Error sending transcription request: {e}")
        # Exit the program if the transcription request is not sent
        exit()

    # Output the transcription in the console
    try:
        print("Transcript: ")
        print(transcript)
    except Exception as e:
        print(f"System: Error printing transcription: {e}")
        # Continue the program even if the transcription is not printed

    # Save the transcription to a text file
    try:
        with open("transcript.txt", "w") as text_file:
            text_file.write(transcript)
            print("System: Transcription saved to file successfully.")
    except Exception as e:
        print(f"System: Error saving transcription to file: {e}")
        # Exit the program if the transcription is not saved to a file
        exit()

except Exception as e:
    print(f"System: An error occurred: {e}")
    # Exit the program if an error occurs
    exit()
