import asyncio
from cls import Controller


async def main():
    # await Controller(input_device='MacBook Air Microphone').start()
    await Controller(input_device='AirPods — Kenny R').start()


asyncio.run(main())
