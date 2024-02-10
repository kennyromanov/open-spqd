import asyncio
from cls import View


async def main():
    await View().say('Hello world')


asyncio.run(main())
