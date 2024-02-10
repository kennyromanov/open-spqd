import asyncio
from cls import Controller


async def main():
    await Controller(input_device='MacBook Air Microphone').start()


asyncio.run(main())
