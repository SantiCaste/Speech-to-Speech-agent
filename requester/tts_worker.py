from threading import Thread
import pyttsx3
from queue import Queue

textQueue = Queue()

def my_tts_worker():
    tts_engine = pyttsx3.init()

    while True:
        text = textQueue.get()
        if text is None:
            textQueue.task_done()
            break
        tts_engine.say(text)
        tts_engine.runAndWait()
        textQueue.task_done()
    
tts_thread = Thread(target=my_tts_worker)
tts_thread.start()

if __name__ == "__main__":
    try:
        while True:
            text = input("Enter text to speak: ")
            textQueue.put(text)
            if text == "exit":
                break
    except KeyboardInterrupt:
        print("\nTerminating...")
    
    textQueue.put(None)
    textQueue.join()
    tts_thread.join()