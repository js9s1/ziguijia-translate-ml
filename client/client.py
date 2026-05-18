import requests
import json

# Server URL
SERVER_URL = "http://127.0.0.1:5600"

def synthesize_text(text, output_file="output.wav", use_json=True):
    """
    Send text to server and save the returned WAV file
    
    Args:
        text: Text to synthesize
        output_file: Path to save the WAV file
        use_json: True for JSON, False for form data
    """
    if use_json:
        # JSON endpoint
        url = f"{SERVER_URL}/synthesize"
        headers = {'Content-Type': 'application/json'}
        data = {
            'text': text,
            'filename': output_file,
        }
        response = requests.post(url, headers=headers, json=data)
    else:
        # Form endpoint
        url = f"{SERVER_URL}/synthesize/form"
        data = {
            'text': text,
            'filename': output_file,
        }
        response = requests.post(url, data=data)
    
    if response.status_code == 200:
        # Save the WAV file
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"WAV file saved as: {output_file}")
        return True
    else:
        print(f"Error: {response.json()}")
        return False

def check_server_health():
    """Check if server is running"""
    try:
        response = requests.get(f"{SERVER_URL}/health")
        if response.status_code == 200:
            print("Server is healthy!")
            return True
    except:
        print("Server is not reachable")
        return False

if __name__ == "__main__":
    # Check server health
    if not check_server_health():
        print("Please make sure the server is running first!")
        exit(1)
    
    # Example 1: Using JSON
    synthesize_text("Hello, this is a test!", "hello_test.wav")
    
    # Example 2: Different text
    synthesize_text("This is another example", "example.wav", use_json=False)
