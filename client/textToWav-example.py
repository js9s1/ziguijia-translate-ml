#!/usr/bin/env python3
"""
Example script showing how to use the Text-to-WAV client library
"""

import io
import wave
from textToWave import TextToWavClient, text_to_wav, text_to_wav_file


def main():
    # Method 1: Using the convenience functions
    print("Method 1: Using convenience functions")
    
    # Get WAV data as bytes
    wav_bytes = text_to_wav("Hello, this is a test message.")
    print(f"Received {len(wav_bytes)} bytes of WAV data")
    
    # Save to file
    text_to_wav_file("This is another test.", "output_test.wav")
    print("Saved to output_test.wav")
    
    # Method 2: Using the client class
    print("\nMethod 2: Using client class")
    client = TextToWavClient(host='127.0.0.1', port=5600, timeout=30)
    
    # Get as bytes
    wav_data = client.text_to_wav("This is from the client class.")
    print(f"Received {len(wav_data)} bytes")
    
    # Save to file
    client.text_to_wav_file("Saving directly to file.", "client_output.wav")
    
    # Get as stream
    wav_stream = client.text_to_wav_stream("Streaming example.")
    print(f"Stream created with {wav_stream.getbuffer().nbytes} bytes")
    
    # You can even process the WAV stream
    wav_stream.seek(0)
    with wave.open(wav_stream, 'rb') as wav_file:
        print(f"WAV parameters: {wav_file.getparams()}")
    
    # Batch processing example
    print("\nBatch processing example:")
    texts = [
        "First message",
        "Second message",
        "Third message"
    ]
    
    for i, text in enumerate(texts, 1):
        wav_data = client.text_to_wav(text)
        print(f"Generated WAV {i}: {len(wav_data)} bytes")


if __name__ == "__main__":
    main()