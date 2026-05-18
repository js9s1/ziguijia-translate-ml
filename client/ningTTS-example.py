#!/usr/bin/env python3
"""
Example script demonstrating how to use the TTS client library.
"""

from ningTTS import TTSClient
import numpy as np
import sounddevice as sd  # Optional: for playing audio

def main():
    # Initialize the client
    client = TTSClient("http://127.0.0.1:5600")
    
    # Check if server is healthy
    if not client.health_check():
        print("Server is not healthy. Make sure it's running.")
        return
    
    # Example 1: Get WAV bytes
    print("Example 1: Getting WAV bytes")
    text = "Hello, this is a test of the text to speech system."
    audio_bytes = client.synthesize(text)
    print(f"Received {len(audio_bytes)} bytes of audio data")
    
    # Example 2: Save to file
    print("\nExample 2: Saving to file")
    client.synthesize_to_file("Hello, world!", "output.wav")
    print("Audio saved to output.wav")
    
    # Example 3: Get as numpy array (for further processing)
    print("\nExample 3: Getting numpy array")
    audio_array = client.synthesize_to_numpy("This is a test")
    print(f"Audio array shape: {audio_array.shape}, dtype: {audio_array.dtype}")
    
    # Optional: Play the audio (requires sounddevice)
    try:
        print("\nPlaying audio...")
        sd.play(audio_array, samplerate=44100)
        sd.wait()
    except:
        print("Could not play audio (sounddevice not installed)")
    
    # Example 4: With voice parameters (if supported by server)
    print("\nExample 4: With voice parameters")
    voice_params = {
        'voice': 'female',
        'speed': 1.2,
        'pitch': 1.1
    }
    client.synthesize_to_file(
        "This is with custom voice parameters.",
        "custom_voice.wav",
        voice_params
    )
    print("Audio with custom parameters saved")

if __name__ == "__main__":
    main()
