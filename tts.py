import sys, audioop
sys.modules["pyaudioop"] = audioop
from gtts import gTTS
import time
from pydub import AudioSegment
import nltk

def text_to_speech(text, lang='en-uk', tld='co.uk', chunk_pause_ms=200):
    # 1. Split into sentences
    sentences = nltk.sent_tokenize(text)
    parts = []
    for sentence in sentences:
        tts = gTTS(text=sentence, lang=lang, tld=tld, slow=False)
        part_path = f"temp_{int(time.time()*1000)}.mp3"
        tts.save(part_path)
        part_audio = AudioSegment.from_file(part_path)
        parts.append(part_audio)
    # 2. Combine with pauses
    combined = AudioSegment.silent(duration=0)
    for part in parts:
        combined += part + AudioSegment.silent(duration=chunk_pause_ms)
    # 3. Optional: apply slight tempo change, filter
    combined = combined.speedup(playback_speed=0.95)
    combined = combined.low_pass_filter(6000)
    # 4. Output
    output_path = f"{int(time.time())}.mp3"
    combined.export(output_path, format="mp3")
    return output_path

# Usage
output = text_to_speech("Hello there! This is a test of gTTS with a UK accent.")
print("Audio saved at:", output)
