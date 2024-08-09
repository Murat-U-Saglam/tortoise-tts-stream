import asyncio
from asyncio import Queue
import logging
import multiprocessing
import socket
import numpy as np
import sounddevice as sd
from functools import partial
from ollama import AsyncClient
import time
import asyncio

import multiprocessing

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def play_audio_stream(client_socket):
    buffer = b""
    stream = sd.OutputStream(samplerate=24000, channels=1, dtype="float32")
    stream.start()

    try:
        while True:
            chunk = client_socket.recv(1024)
            if b"END_OF_AUDIO" in chunk:
                buffer += chunk.replace(b"END_OF_AUDIO", b"")
                if buffer:
                    audio_array = np.frombuffer(buffer, dtype=np.float32)
                    stream.write(audio_array)
                break

            buffer += chunk
            while len(buffer) >= 4096:
                audio_chunk = buffer[:4096]
                audio_array = np.frombuffer(audio_chunk, dtype=np.float32)
                stream.write(audio_array)
                buffer = buffer[4096:]

    finally:
        stream.stop()
        stream.close()


async def send_text_to_server(text):
    server_ip = "localhost"
    server_port = 5000
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((server_ip, server_port))

    logging.info(f"Sending text to server: {text}")
    character_name = "deniro"
    try:
        data = f"{character_name}|{text}"
        client_socket.sendall(data.encode("utf-8"))

        await play_audio_stream(client_socket)

        print("Audio playback finished.")

    finally:
        client_socket.close()


async def chat():
    """
    Stream a chat from Llama using the AsyncClient.
    """
    message = {"role": "user", "content": "Tell me an interesting fact about elephants"}
    async for part in await AsyncClient().chat(
        model="llama3", messages=[message], stream=True
    ):
        tmp_text = part["message"]["content"]
        yield tmp_text


async def process_stream(queue: Queue, batch_size=10):
    batch = []  # Initialize an empty batch
    async for text in chat():
        batch.append(text)
        if len(batch) >= batch_size:
            logging.info(f"Processing batch: {batch}")
            await queue.put(batch)
            batch = []

    await queue.put(None)  # Signal that processing is complete
    logging.info("Processing complete")


async def narrate(queue: Queue, client_socket=None):
    """
    Narrate the processed batches from the queue.
    """

    while True:
        processed_batch = await queue.get()
        if processed_batch is None:
            break
        await send_text_to_server(text=processed_batch)
        logging.info(f"Narrated batch. Remaining in queue: {queue.qsize()}")

    logging.info("Narration complete")


async def main():
    queue = Queue()

    # Create tasks for processing and narration
    process_task = asyncio.create_task(process_stream(queue))
    narrate_task = asyncio.create_task(narrate(queue))

    # Wait for both tasks to complete
    await asyncio.gather(process_task, narrate_task)


if __name__ == "__main__":
    # Run the server in a different terminal command python tortoise/tts_stream.py
    asyncio.run(main())
