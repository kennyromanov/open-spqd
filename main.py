#!/usr/bin/env ./venv/bin/python3

"""
Open SpeakQ Service 1
by Kenny Romanov
"""

import asyncio
import fwk
from argparse import ArgumentParser
from cls import Assistant

parser = ArgumentParser()
parser.add_argument("--voice", default="alloy")
parser.add_argument("--sensitivity", default="60")
parser.add_argument("--input", default="_default")
parser.add_argument("--output", default="_default")
args = parser.parse_args()


async def main() -> None:
    tts_voice = str(args.voice)
    input_device = str(args.input)
    input_sensitivity = int(args.sensitivity)
    output_device = str(args.output)

    if input_device == '_default':
        input_device = fwk.default_input(True)
    if output_device == '_default':
        output_device = fwk.default_output(True)

    match input_device:
        case '-':
            input_stream = fwk.stdin()
        case _:
            input_stream = fwk.record_audio(input_device)

    match output_device:
        case '-':
            output_stream = fwk.stdout()
        case _:
            output_stream = fwk.play_audio(output_device, 48000, 1)

    try:
        await Assistant(input_stream, output_stream).start()
        await input_stream.coroutine
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass


asyncio.run(main())
