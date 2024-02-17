"""
Open SpeakQ Service 1
by Kenny Romanov
"""

import asyncio
from argparse import ArgumentParser
from cls import Controller

# parser = ArgumentParser()
# parser.add_argument("--voice", default="-")
# parser.add_argument("--sensitivity", default="-")
# parser.add_argument("--input", default="-")
# args = parser.parse_args()


async def main():
    await Controller(
        input_device='AirPods — Kenny R',
        input_sensitivity=60,
    ).start()

asyncio.run(main())
