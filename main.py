from langchain_community.llms import Ollama
import whisper, pyttsx3
import sounddevice as sd
import numpy as np
import sys

local_model_path = 'deepseek-r1:32b' 
llm =Ollama(model=local_model_path)

whisper_model = whisper.load_model("base")
tts_engine = pyttsx3.init()

AUDIO_STOPPER_KEYWORD = ['finalizar', 'stop', 'terminar', 'detener']

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
        #print(f'current rms: {rms}. Silence buffer: {len(silence_buffer)}. Audio buffer: {len(audio_buffer)}')
        if rms < silence_threshold and len(audio_buffer) > 45000: #this is done so that the audio is stopped once a certain time is reached and not before that. (min audio) 
            break

    stream.stop()
    stream.close()
    print("Recording complete.")
    audio = np.array(audio_buffer, dtype=np.int16)
    audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    return np.squeeze(audio)        

    """
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Recording complete.")
    audio = audio.astype(np.float32) / np.iinfo(np.int16).max
    return np.squeeze(audio)"
    """

def speech_to_text(audio):
    result = whisper_model.transcribe(audio)
    text = result['text']
    print(f"You said: {text}")
    return text

def text_to_speech(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def parse_response(response: str) -> str:
    parts = response.split('</think>')
    if len(parts) != 2:
        return parts[0]

    return parts[1]

def contains_any(string: str, elements: list) -> bool:
    for element in elements:
        if element in string:
            return True

    return False


def process_input(input_text: str):
    response = llm.invoke(input_text)
    print(response)
    response = parse_response(response) #preprocess the response

    text_to_speech(response)

def main_function(tts=True, stt=True):
    running =  True
    input_text = None

    while running:
        if stt:
            audio = record_audio(silence_threshold=55)
            input_text = speech_to_text(audio)
    
        if not input_text or not stt:
            input_text = input('Type your request:')

        inp = input_text.lower()
        
        if contains_any(inp, AUDIO_STOPPER_KEYWORD):
            running = False
            print("Goodbye!")
            break

        elif 'comenzar' in inp: 
            #remove the word 'comenzar' from the input
            input_text = input_text.replace('comenzar', '', 1)
            #process_input(input_text)

            response = llm.invoke(input_text)
            print(response)

            if tts:
                response = parse_response(response) #preprocess the response
                text_to_speech(response)

        else:
            print(f"No te entendí. Dijiste {input_text}")
 

if __name__ == "__main__":
    #read args from command line
    args = sys.argv[1:]
    print(args)
    if args:
        tts = True if 'tts' in args else False
        stt = True if 'stt' in args else False
        main_function(tts, stt)
    else:
        main_function(False, False)