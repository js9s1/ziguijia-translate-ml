import requests
import io
from typing import Optional, Dict, Any
import wave
import numpy as np

class TTSClient:
    """
    Client library for the Text-to-Speech service.
    """
    
    def __init__(self, base_url: str = "http://localhost:5600" , timeout: int = 360):
        """
        Initialize the TTS client.
        
        Args:
            base_url: The base URL of the TTS server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def synthesize(self, text: str, prompt_file: str = None) -> bytes:
        """
        Convert text to speech and return WAV audio bytes.
        
        Args:
            text: The text to synthesize
            prompt_file: Optional voice parameters (depends on server implementation)
            
        Returns:
            bytes: WAV audio data
        
        Raises:
            TTSException: If the request fails
        """
        try:
            response = self.session.post(
                f"{self.base_url}/synthesize",
                json={
                    'text': text,
                    'prompt_file': prompt_file 
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            # Check if response is audio
            content_type = response.headers.get('content-type', '')
            if 'audio/wav' in content_type:
                return response.content
            else:
                # Handle error response
                error_data = response.json()
                raise TTSException(f"Server error: {error_data.get('error', 'Unknown error')}")
                
        except requests.exceptions.RequestException as e:
            raise TTSException(f"Request failed: {str(e)}")
    
    def synthesize_to_file(self, text: str, output_file: str, 
                           prompt_file: str = None) -> None:
        """
        Synthesize text and save to a WAV file.
        
        Args:
            text: The text to synthesize
            output_file: Path to save the WAV file
            prompt_file: Optional voice parameters
        """
        audio_data = self.synthesize(text, prompt_file)
        with open(output_file, 'wb') as f:
            f.write(audio_data)
    
    def synthesize_to_numpy(self, text: str, 
                            prompt_file: str = None) -> np.ndarray:
        """
        Synthesize text and return as numpy array.
        
        Args:
            text: The text to synthesize
            prompt_file: Optional voice parameters
            
        Returns:
            np.ndarray: Audio data as numpy array
        """
        audio_data = self.synthesize(text, prompt_file)
        
        # Convert WAV bytes to numpy array
        with io.BytesIO(audio_data) as wav_io:
            with wave.open(wav_io, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                dtype = np.int16 if wav_file.getsampwidth() == 2 else np.int32
                audio_array = np.frombuffer(frames, dtype=dtype)
                
        return audio_array
    
    def health_check(self) -> bool:
        """
        Check if the server is healthy.
        
        Returns:
            bool: True if server is healthy, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

class TTSException(Exception):
    """Custom exception for TTS client errors."""
    pass
