import os
import subprocess
import math
import re
import asyncio
import typing
import pydub
import wave
import sounddevice as sd
import numpy as np
import redis
import openai
from io import BytesIO
from typing import List
from dotenv import load_dotenv

# Third-Parties

load_dotenv()

r = redis.Redis()

openai.api_key = os.environ.get("OPENAI_API_KEY")


# Types

GptModel = typing.Literal[
    'gpt-3.5-turbo-0125',
    'gpt-4-turbo-preview',
]

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
    'info',
    'error',
]

AsstDeltaType = typing.Literal[
    'chunk',
    'stop',
]


# Classes

class SpqError(Exception):
    pass


class StreamEvent:
    def __init__(self, event: StreamEvents, handler: typing.Callable) -> None:
        self.event = event
        self.handler = handler

    async def soft(self, event: StreamEvents, data) -> None:
        if self.event != event:
            return
        await self.handler(data)

    async def work(self, data) -> None:
        await self.handler(data)


class StreamBus:
    def __init__(self, name_from: str = "Unknown", name_to: str = "All", data: typing.Any = None) -> None:
        self.name_from = name_from
        self.name_to = name_to
        self.data = data


class Stream:
    def __init__(self, task: typing.Any = None, subs: List[StreamEvent] = None) -> None:
        if subs is None:
            subs = []

        self.coroutine = task
        self.subs = subs

    async def emit(self, event: StreamEvents, data) -> None:
        tasks = [asyncio.create_task(sub.soft(event, data)) for sub in self.subs]
        await asyncio.gather(*tasks)

    async def write(self, value) -> None:
        await self.emit('data', value)

    async def info(self, name_from: str, name_to: str, data: typing.Any) -> None:
        info_bus = StreamBus(name_from, name_to, data)
        await self.emit('info', info_bus)

    async def close(self, value=None) -> None:
        await self.emit('close', value)

    async def error(self, e: BaseException) -> None:
        await self.emit('error', e)

    def task(self, coroutine: typing.Any) -> None:
        self.coroutine = coroutine

    def on(self, event: StreamEvents, callback: typing.Callable) -> None:
        self.subs.append(StreamEvent(event, callback))


class AsstTool:
    def __init__(self, name: str, descr: str, args: object, required: object, handler: typing.Callable) -> None:
        self.name = name
        self.descr = descr
        self.args = args
        self.required = required
        self.handler = handler


class AsstDelta:
    def __init__(self, type: AsstDeltaType = 'chunk', text: str = None, tool: AsstTool = None) -> None:
        self.type = type
        self.text = text
        self.tool = tool


# Base Functions


def error(message: str) -> SpqError:
    return SpqError(message)


def clear_queue(asyncio_queue: asyncio.Queue) -> None:
    while not asyncio_queue.empty():
        try:
            asyncio_queue.get_nowait()
        except asyncio.QueueEmpty:
            break


def record_audio(
        input_device: str | int,
        samplerate: int = 16000,
        num_channels: int = 1
) -> subprocess.Popen:
    command = [
        'ffmpeg',
        '-f', 'avfoundation',
        '-i', f':{input_device}',
        '-ac', f'{num_channels}',
        '-ar', f'{samplerate}',
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-'
    ]

    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def audio_default_output(strict: bool = False) -> typing.Any:
    try:
        device = sd.query_devices(kind='output')
        return device['index']
    except Exception as e:
        if strict:
            raise e
        return None


def play_audio(input_bytes: bytes, format_name: str, samplerate=48000, output_device: str | int = None) -> None:
    if not output_device:
        output_device = audio_default_output()

    # Decoding the audio
    bytes_pcm = to_pcm(input_bytes, format_name)
    array_pcm = np.frombuffer(bytes_pcm, dtype=np.int32)

    sd.play(data=array_pcm, samplerate=samplerate, device=output_device)
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


def pcm_to_wav(input_bytes: bytes, samplerate: int, num_channels: int) -> BytesIO:
    wav_io = BytesIO()

    # Encoding the audio
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(samplerate)
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
        for chunk in response.response.iter_bytes():
            await audio_stream.write(chunk)

        await audio_stream.close()

    # Instantly returning the stream
    fetching_task = asyncio.create_task(fetch_audio())
    audio_stream.task(fetching_task)

    return audio_stream


def stt(model: SttModel, input_wav: BytesIO, prompt: str = 'Обычная речь, разделенная запятыми.') -> str:
    # Querying the API
    response = openai.audio.transcriptions.create(
        model=model,
        file=input_wav,
        prompt=prompt,
    )

    return response.text


def gpt(model: GptModel, conv: typing.Any) -> Stream:
    answer_stream = Stream()

    async def on_close(value):
        # TODO: kr: Make an ability to stop answering
        pass

    async def fetch_answer():
        # Querying the API
        response = openai.chat.completions.create(
            model=model,
            messages=conv,
            stream=True,
        )

        # Streaming the answer
        for chunk in response:
            await answer_stream.write(chunk.choices[0].delta)

        await answer_stream.close()

    fetching_task = asyncio.create_task(fetch_answer())

    # Instantly returning the stream
    answer_stream.task(fetching_task)
    answer_stream.on('close', on_close)

    return answer_stream


def asst(message: str, funcs: List[AsstTool] = None) -> Stream:
    if not funcs:
        funcs = []

    final_role = ''
    final_content = ''
    final_tools = ''
    answering_stream = Stream()
    answering_buffer = ''

    # Commits the sentences
    async def commit(input_text: str):
        nonlocal answering_buffer

        asst_delta = AsstDelta(
            type='chunk',
            text=input_text
        )

        await answering_stream.write(asst_delta)
        answering_buffer = ''

    # Analyzes the bot output searching for the quantizable chunks
    async def analyze(input_text: str):
        quantisation = 2
        regex = re.compile(r'.+?[.;?!\n]+', flags=re.MULTILINE)
        matches = regex.findall(input_text)

        if len(matches) >= quantisation:
            await commit(input_text)

    async def gpt_on_data(delta) -> None:
        nonlocal final_role, final_content, final_tools, answering_buffer
        if delta.role:
            final_role += delta.role
        if delta.content:
            final_content += delta.content
            answering_buffer += delta.content
        if delta.tool_calls:
            final_tools += delta.tool_calls

        await analyze(answering_buffer)

    async def gpt_on_close(value) -> None:
        nonlocal answering_buffer

        # Committing the rest of the buffer if it is remained
        if answering_buffer:
            await commit(answering_buffer)

        asst_delta = AsstDelta(
            type='stop',
            text=final_content,
            tool=None,
        )

        await answering_stream.write(asst_delta)

    async def gpt_on_error(e) -> None:
        await answering_stream.error(e)

    gpt_stream = gpt('gpt-3.5-turbo-0125', [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': message},
    ])
    gpt_stream.on('data', gpt_on_data)
    gpt_stream.on('close', gpt_on_close)
    gpt_stream.on('error', gpt_on_error)

    answering_stream.task(gpt_stream.coroutine)

    return answering_stream
