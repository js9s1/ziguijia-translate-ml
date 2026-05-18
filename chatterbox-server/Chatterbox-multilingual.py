import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxMultilingualTTS

AUDIO_PROMPT_PATH = "/home/js9s/子归家/51608/ziguijia_51608_audio.wav"

model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
model.prepare_conditionals(AUDIO_PROMPT_PATH)

texts = [
    "Hi there, Sarah here from MochaFone calling you back [chuckle], have you got one minute to chat about the billing issue?",
    "This is a second text that will use the same voice prompt.",
    "And a third text for good measure.",
]

for i, text in enumerate(texts):
    wav = model.generate(text, language_id="en")
    ta.save(f"test-multilingual-{i}.wav", wav, model.sr)
