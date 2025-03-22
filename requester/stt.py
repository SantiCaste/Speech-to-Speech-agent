import whisper
import numpy as np
import sounddevice as sd

AUDIO_STOPPER_KEYWORD = ['finalizar', 'stop', 'terminar', 'detener', 'fin', 'gracias']

whisper_model = whisper.load_model("base")

def record_audio(fs=16000, silence_threshold=0.01, silence_duration=1.0):
    print("Recording audio...")
    audio_buffer = []
    silence_buffer = []
    stream = sd.InputStream(samplerate=fs, channels=1, dtype='int16')
    stream.start()

    while True:
        audio_chunk, _ = stream.read(fs // 10)  # Read in chunks of 100ms
        audio_buffer.extend(audio_chunk)
        silence_buffer.extend(audio_chunk)

        # Check if the silence buffer exceeds the silence duration
        if len(silence_buffer) > fs * silence_duration:
            silence_buffer = silence_buffer[-int(fs * silence_duration):]

        # Calculate the RMS value of the silence buffer
        rms = np.sqrt(np.mean(np.square(np.array(silence_buffer, dtype=np.float32))))
        if rms < silence_threshold and len(audio_buffer) > 45000: #this is done so that the audio is stopped once a certain time is reached and not before that. (min audio) 
            break

    stream.stop()
    stream.close()
    print("Recording complete.")
    audio = np.array(audio_buffer, dtype=np.int16)
    audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    return np.squeeze(audio)        

def speech_to_text(audio):
    result = whisper_model.transcribe(audio)
    text = result['text']
    print(f"You said: {text}")
    return text
