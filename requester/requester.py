import requests
import json
import tts_worker, stt

model = 'deepseek-r1:32b'

TOKENS_PER_SECOND = 39

def generate(prompt, context):
    prePrompt = "Keep the response at most 50 words long. The prompt is: "
    prompt = prePrompt + prompt
    r = requests.post('http://localhost:11434/api/generate',
                    json={
                        'model': model,
                        'prompt': prompt,
                        'context': context
                    },
                    stream=True)
    r.raise_for_status()

    buffer = []
    hasFinishedThinking: bool = False

    for line in r.iter_lines():
        body: dict = json.loads(line)
        response_part = body.get('response', '')

        # Wait until the 'finished thinking' flag is set
        if not hasFinishedThinking and response_part != '</think>':
            continue
        elif response_part == '</think>': #this is only done once (after the previous if is false) and can be optimized by removing the if
            hasFinishedThinking = True
            continue
        
        buffer.append(response_part)

        # Send the accumulated responses to the TTS worker when the buffer size is reached
        if '.' in response_part or body.get('done', False): # so that sentences are not cut in the middle
            processed_buffer = ' '.join(buffer)
            tts_worker.textQueue.put(processed_buffer)
            buffer = []

        if 'error' in body:
            raise Exception(body['error'])

        if body.get('done', False):
            return body['context']

def main(sttMode=True):
    running = True
    input_text = None
    context = []
    started = False

    try:
        while running:
            if not tts_worker.textQueue.empty(): #to wait for the tts to finish speaking
                continue
            
            if sttMode:
                audio = stt.record_audio(silence_threshold=55)
                input_text = stt.speech_to_text(audio)

            if not input_text or not sttMode:
                input_text = input("You: ")

            inp = input_text.lower()

            if any(keyword in inp for keyword in stt.AUDIO_STOPPER_KEYWORD):
                running = False
                print("Goodbye!")
                break

            if not started:
                if 'comenzar' in inp:
                    started = True
                    print("Starting conversation...")
                    inp = inp.replace('comenzar', '', 1)
                else:
                    print("Say 'comenzar' to start the conversation.")
                    continue

            context = generate(inp, context)

    except KeyboardInterrupt:
        print("\nTerminating...")

    tts_worker.textQueue.put(None)
    tts_worker.textQueue.join()
    tts_worker.tts_thread.join()

if __name__ == '__main__':
    main()