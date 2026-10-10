"""Terminal Loan Assistant: type your question, hear the answer."""

import asyncio

import sounddevice as sd

# Reuse your existing agent setup (runner, greeting, history helper)
from server import runner, GREETING_PROMPT, extract_messages

SAMPLE_RATE = 24000


async def main():
    session = await runner.run()
    audio_queue: asyncio.Queue = asyncio.Queue()
    printed_ids = set()
    latest_history = []

    speaker = sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    speaker.start()

    async def play_audio():
        """Play queued audio chunks without blocking the event loop."""
        while True:
            chunk = await audio_queue.get()
            await asyncio.to_thread(speaker.write, chunk)

    def clear_audio():
        while not audio_queue.empty():
            audio_queue.get_nowait()

    def print_new_replies():
        for m in extract_messages(latest_history):
            if m["role"] == "assistant" and m["id"] not in printed_ids:
                printed_ids.add(m["id"])
                print(f"\nAssistant: {m['text']}")

    async def listen_to_agent():
        async for event in session:
            if event.type == "audio":
                await audio_queue.put(event.audio.data)
            elif event.type == "audio_interrupted":
                clear_audio()
            elif event.type == "history_updated":
                latest_history[:] = list(event.history)
            elif event.type == "agent_end":
                print_new_replies()
            elif event.type == "guardrail_tripped":
                clear_audio()
                print("\n[Off-topic reply blocked - I can only help with loan questions.]")
            elif event.type == "error":
                print(f"\n[Error: {event.error}]")

    async with session:
        player_task = asyncio.create_task(play_audio())
        listener_task = asyncio.create_task(listen_to_agent())

        await session.send_message(GREETING_PROMPT)  # agent speaks first
        print("Loan Assistant ready. Type your question, or 'quit' to exit.")

        try:
            while True:
                text = await asyncio.to_thread(input, "\nYou: ")
                if text.strip().lower() in {"quit", "exit"}:
                    break
                if text.strip():
                    await session.send_message(text)
        finally:
            player_task.cancel()
            listener_task.cancel()
            speaker.stop()
            speaker.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass