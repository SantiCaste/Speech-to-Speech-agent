import pyttsx3
import requests
import json
import tts_worker 

model = 'deepseek-r1:32b'


def generate(prompt, context):
    r = requests.post('http://localhost:11434/api/generate',
                    json={
                        'model': model,
                        'prompt': prompt,
                        'context': context
                    },
                    stream=True)
    r.raise_for_status()

    for line in r.iter_lines():
        body: dict = json.loads(line)
        response_part = body.get('response', '')

        tts_worker.textQueue.put(response_part) #this goes word by word, it's too slow. I need to change this so that i add more than one word at a time

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
            if user_input == "exit":
                break

            context = generate(user_input, context)

    except KeyboardInterrupt:
        print("\nTerminating...")
    
    tts_worker.textQueue.put(None)
    tts_worker.textQueue.join()
    tts_worker.tts_thread.join()
           
if __name__ == '__main__':
    main()