import os
import subprocess
import math
import numpy
import asyncio
import typing
import pydub
import wave
import sounddevice as sd
import colorama
import redis
import openai
import traceback
from io import BytesIO
from typing import List
from colorama import Fore, Style
from dotenv import load_dotenv
from argparse import ArgumentParser

# Third-Parties

parser = ArgumentParser()
parser.add_argument("--voice", default="-")
parser.add_argument("--sensitivity", default="-")
parser.add_argument("--input", default="-")
args = parser.parse_args()

load_dotenv()

colorama.init()

r = redis.Redis()

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Types

TtsModel = typing.Literal[
    'tts-1',
    'tts-1-hd',
]

TtsVoice = typing.Literal[
    'alloy',
    'echo',
    'fable',
    'onyx',
    'nova',
    'shimmer',
]

SttModel = typing.Literal[
    'whisper-1',
]

StreamEvents = typing.Literal[
    'open',
    'data',
    'close',
    'error',
]


# Classes

class SpqError(Exception):
    pass


class StreamEvent:
    def __init__(self, event: StreamEvents, handler: typing.Callable):
        self.event = event
        self.handler = handler

    async def soft(self, event: StreamEvents, data) -> None:
        if self.event != event:
            return
        await self.handler(data)

    async def work(self, data) -> None:
        await self.handler(data)


class Stream:
    def __init__(self, task: typing.Any = None, subs: List[StreamEvent] = None):
        if subs is None:
            subs = []

        self.task_coroutine = task
        self.subs = subs

    async def emit(self, event: StreamEvents, data) -> None:
        tasks = [asyncio.create_task(sub.soft(event, data)) for sub in self.subs]
        await asyncio.gather(*tasks)

    async def write(self, value) -> None:
        await self.emit('data', value)

    async def close(self, value=None) -> None:
        await self.emit('close', value)

    async def error(self, e: BaseException) -> None:
        await self.emit('error', e)

    def task(self, coroutine: typing.Any) -> None:
        self.task_coroutine = coroutine

    def on(self, event: StreamEvents, callback: typing.Callable) -> None:
        self.subs.append(StreamEvent(event, callback))


# Base Functions

def beautify_error(e: Exception) -> str:
    return f"\n\n{Fore.RED + Style.BRIGHT}(!) Open Chatter Error\n{Fore.RESET + str(e)}\n"


def beautify_log(message: str) -> str:
    return f"\n{Fore.GREEN}(+) Open Chatter: {message}"


def log(message: str) -> None:
    print(beautify_log(message))


def log_error(e: Exception) -> None:
    print(beautify_error(e))


def default_output(strict: bool = False) -> typing.Any:
    try:
        device = sd.query_devices(kind='output')
        return device
    except Exception as e:
        if strict:
            raise e
        return None


def record_audio(
        input_device: str | int,
        frame_rate: int = 16000,
        num_channels: int = 1
) -> subprocess.Popen:
    command = [
        'ffmpeg',
        '-f', 'avfoundation',
        '-i', f':{input_device}',
        '-ac', f'{num_channels}',
        '-ar', f'{frame_rate}',
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-'
    ]

    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def play_audio(input_bytes: bytes, format_name: str, output_device: str | int = None) -> None:
    if not output_device:
        output_device = default_output()['index']

    # Decoding the audio
    bytes_pcm = to_pcm(input_bytes, format_name)
    array_pcm = numpy.frombuffer(bytes_pcm, dtype=numpy.int32)

    sd.play(array_pcm, samplerate=48000, device=output_device)
    sd.wait()


def calc_volume(buffer: bytes) -> float:
    calc_sum = 0
    sample_count = 0

    # Reading the 16-bit values from the buffer
    for i in range(0, len(buffer), 2):
        if i + 1 < len(buffer):
            int16 = int.from_bytes(buffer[i:i + 2], byteorder='little', signed=True)
            calc_sum += int16 ** 2
            sample_count += 1

    if sample_count == 0:
        return 0

    rms = math.sqrt(calc_sum / sample_count)
    db = 20 * math.log10((rms + 1e-9) / 32768)

    min_db = -60.0
    max_db = 20.0
    result = (db - min_db) / (max_db - min_db) * 100

    # Trimming the value by 0 and 100
    result = 100 if result > 100 else result
    result = 0 if result < 0 else result

    return result


def to_pcm(input_bytes: bytes, format_name: str) -> bytes:
    file_ogg = BytesIO(input_bytes)
    audio = pydub.AudioSegment.from_file(file_ogg, format=format_name)
    bytes_pcm = audio.raw_data
    return bytes_pcm


def pcm_to_wav(input_bytes: bytes, frame_rate: int, num_channels: int) -> BytesIO:
    wav_io = BytesIO()

    # Encoding the audio
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(input_bytes)
    wav_io.seek(0)

    return wav_io


# Framework Functions

def tts(model: TtsModel, voice: TtsVoice, input_text: str) -> Stream:
    audio_stream = Stream()

    async def fetch_audio():
        nonlocal model, voice, input_text, audio_stream

        # Querying the API
        response = openai.audio.speech.create(
            model=model,
            voice=voice,
            input=input_text,
            response_format="opus",
        )

        # Streaming the audio
        for data in response.response.iter_bytes():
            await audio_stream.write(data)

        await audio_stream.close()

    fetching_task = asyncio.create_task(fetch_audio())
    audio_stream.task(fetching_task)

    return audio_stream


def stt(model: SttModel, filename: str, prompt: str = 'Обычная речь, разделенная запятыми.') -> str:
    audio_file = open(filename, "rb")

    response = openai.audio.transcriptions.create(
        model=model,
        file=audio_file,
        prompt=prompt,
    )

    return response.text
