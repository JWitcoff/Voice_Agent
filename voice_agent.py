from openai import OpenAI
from dotenv import load_dotenv
from agents import Agent, function_tool, WebSearchTool, FileSearchTool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents import Runner, trace
import os

# --- Voice Agent Imports---
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline

# Load Environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
vector_store_id = {"id": os.getenv("VECTOR_STORE_ID")}



# --- Tool 1: Fetch account information (dummy) ---
@function_tool
def get_account_info(user_id: str) -> dict:
    return {
        "user_id": user_id,
        "name": "Justin",
        "account_balance": "$100,000",
        "membership_status": "Principal"
    }

# --- Agent Definitions ---

# --- Agent 1: Search Agent ---
search_agent = Agent(
    name="SearchAgent",
    instructions="You immediately provide an input to the WebSearchTool to find up-to-date information on the user's query.",
    tools=[WebSearchTool()],
)

# --- Agent 2: Knowledge Agent ---
knowledge_agent = Agent(
    name="KnowledgeAgent",
    instructions="You answer user questions on our product portfolio with concise, helpful responses using the FileSearchTool.",
    tools=[FileSearchTool(
        max_num_results=3,
        vector_store_ids=[vector_store_id["id"]],
    )],
)

# --- Agent 3: Account Agent ---
account_agent = Agent(
    name="AccountAgent",
    instructions="You provide account information based on a user ID using the get_account_info tool.",
    tools=[get_account_info],
)

# --- Agent 4: Triage Agent ---
triage_agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions("""
You are the virtual assistant for Kodawari Labs. Welcome the user and ask how you can help.
Based on the user's intent, route to:
- AccountAgent for account-related queries
- KnowledgeAgent for product FAQs
- SearchAgent for anything requiring real-time web search
"""),
    handoffs=[account_agent, knowledge_agent, search_agent],
)

# -- Voice Assistant -- 
async def voice_assistant():
    samplerate = sd.query_devices(kind='input')['default_samplerate']

    while True:
        pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(triage_agent))

        # Check for input to either provide voice or exit
        cmd = input("Press Enter to speak your query (or type 'esc' to exit): ")
        if cmd.lower() == "esc":
            print("Exiting...")
            break      
        print("Listening...")
        recorded_chunks = []

         # Start streaming from microphone until Enter is pressed
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16', callback=lambda indata, frames, time, status: recorded_chunks.append(indata.copy())):
            input()

        # Concatenate chunks into single buffer
        recording = np.concatenate(recorded_chunks, axis=0)

        # Input the buffer and await the result
        audio_input = AudioInput(buffer=recording)

        with trace("ACME App Voice Assistant"):
            result = await pipeline.run(audio_input)

         # Transfer the streamed result into chunks of audio
        response_chunks = []
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                response_chunks.append(event.data)

        response_audio = np.concatenate(response_chunks, axis=0)

        # Play response
        print("Assistant is responding...")
        sd.play(response_audio, samplerate=samplerate)
        sd.wait()
        print("---")

# --- Test Queries ---
async def test_queries():
    examples = [
        "What's my Kodawari Labs balance? My user ID is 1234567890",
        "Ok, great. We have some breathing room. What's the latest on the AI-powered personal assistants?",
        "Hmmm, what about new AI software - what's trending right now?",
    ]

    samplerate = sd.query_devices(kind='output')['default_samplerate']

    for i, query in enumerate(examples):
        print(f"User: {query}")
        pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(triage_agent))
        audio_input = AudioInput.from_text(query)

        with trace(f"Test Run {i+1}"):
            result = await pipeline.run(audio_input)

        response_chunks = []
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                response_chunks.append(event.data)

        response_audio = np.concatenate(response_chunks, axis=0)

        # Save to .wav file
        filename = f"voice_agents_audio/response_{i+1}.wav"
        os.makedirs("voice_agents_audio", exist_ok=True)
        write(filename, int(samplerate), response_audio)
        print(f"Saved response to: {filename}")
        print("---")

# Run the voice assistant
if __name__ == "__main__":
    import asyncio
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "voice"

    if mode == "test":
        asyncio.run(test_queries())
    else:
        asyncio.run(voice_assistant())

