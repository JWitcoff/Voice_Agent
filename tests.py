# test_queries.py
from voice_agent import triage_agent, Runner, trace
import asyncio

async def test_queries():
    examples = [
        "What's my Kodawari Labs balance? My user ID is 1234567890",
        "Ok, great. We have some breathing room. What's the latest on the AI-powered personal assistants?",
        "Hmmm, what about new AI software - what's trending right now?",
    ]
    with trace("Kodawari Labs Assistant"):
        for query in examples:
            result = await Runner.run(triage_agent, query)
            print(f"User: {query}")
            print(result.final_output)
            print("---")

if __name__ == "__main__":
    asyncio.run(test_queries())
