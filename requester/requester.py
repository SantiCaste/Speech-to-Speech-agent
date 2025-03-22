import pyttsx3
import requests
import json
import tts_worker

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
    buffer_size = TOKENS_PER_SECOND  # Adjust this value as needed

    hasFinishedThinking: bool = False

    for line in r.iter_lines():
        body: dict = json.loads(line)
        response_part = body.get('response', '')

        # Wait until the 'finished thinking' flag is set
        if not hasFinishedThinking and response_part != '</think>':
            continue
        
        hasFinishedThinking = True

        buffer.append(response_part)

        # Send the accumulated responses to the TTS worker when the buffer size is reached
        if len(buffer) >= buffer_size or body.get('done', False):
            tts_worker.textQueue.put(' '.join(buffer))
            buffer = []

        if 'error' in body:
            raise Exception(body['error'])

        if body.get('done', False):
            return body['context']

def main():
    context = []

    try:
        while True:
            user_input = input('You: ')
            if not user_input:
                break
            if user_input.lower() == "exit":
                break

            context = generate(user_input, context)

    except KeyboardInterrupt:
        print("\nTerminating...")

    tts_worker.textQueue.put(None)
    tts_worker.textQueue.join()
    tts_worker.tts_thread.join()

if __name__ == '__main__':
    main()