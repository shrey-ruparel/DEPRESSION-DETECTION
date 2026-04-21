# Alag test file banao — test_mic.py
import sounddevice as sd
import soundfile as sf

print("Recording 5 seconds...")
audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()
sf.write("test_output.wav", audio, 16000)
print("Saved to test_output.wav — play karke suno")