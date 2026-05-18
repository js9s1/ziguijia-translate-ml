import socket
import struct
import io
import wave
from typing import Optional, Union, BinaryIO


class TextToWavClient:
    """
    Client library for Text-to-WAV server
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 5600, timeout: int = 30):
        """
        Initialize the client
        
        Args:
            host: Server hostname or IP
            port: Server port
            timeout: Socket timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
    
    def text_to_wav(self, text: str) -> bytes:
        """
        Convert text to WAV data by calling the server
        
        Args:
            text: Input text to convert
        
        Returns:
            WAV data as bytes
        
        Raises:
            ConnectionError: If connection to server fails
            RuntimeError: If server returns an error
        """
        # Create socket and connect to server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        
        try:
            sock.connect((self.host, self.port))
            
            # Send text length and text data
            text_data = text.encode('utf-8')
            text_length = len(text_data)
            
            # Pack text length as 4-byte unsigned integer (big-endian)
            sock.sendall(struct.pack('>I', text_length))
            sock.sendall(text_data)
            
            # Receive response: status (1 byte) + data length (4 bytes)
            raw_response_header = self._recv_all(sock, 5)
            if not raw_response_header:
                raise ConnectionError("Failed to receive response header")
            
            status, data_length = struct.unpack('>BI', raw_response_header)
            
            # Receive the actual data
            data = self._recv_all(sock, data_length)
            if not data:
                raise ConnectionError("Failed to receive data")
            
            if status == 1:  # Error
                error_message = data.decode('utf-8')
                raise RuntimeError(f"Server error: {error_message}")
            
            return data
            
        finally:
            sock.close()
    
    def text_to_wav_file(self, text: str, output_file: Union[str, BinaryIO]) -> None:
        """
        Convert text to WAV and save to file
        
        Args:
            text: Input text to convert
            output_file: Output file path or file-like object
        """
        wav_data = self.text_to_wav(text)
        
        if isinstance(output_file, str):
            with open(output_file, 'wb') as f:
                f.write(wav_data)
        else:
            output_file.write(wav_data)
    
    def text_to_wav_stream(self, text: str) -> io.BytesIO:
        """
        Convert text to WAV and return as BytesIO stream
        
        Args:
            text: Input text to convert
        
        Returns:
            BytesIO object containing WAV data
        """
        wav_data = self.text_to_wav(text)
        return io.BytesIO(wav_data)
    
    def _recv_all(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """Helper method to receive exactly n bytes"""
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)


# Convenience functions for easy import
def text_to_wav(text: str, host: str = 'localhost', port: int = 8888) -> bytes:
    """
    Convenience function to convert text to WAV
    
    Args:
        text: Input text to convert
        host: Server hostname
        port: Server port
    
    Returns:
        WAV data as bytes
    """
    client = TextToWavClient(host, port)
    return client.text_to_wav(text)


def text_to_wav_file(text: str, output_file: Union[str, BinaryIO], 
                     host: str = 'localhost', port: int = 8888) -> None:
    """
    Convenience function to convert text to WAV file
    
    Args:
        text: Input text to convert
        output_file: Output file path or file-like object
        host: Server hostname
        port: Server port
    """
    client = TextToWavClient(host, port)
    client.text_to_wav_file(text, output_file)