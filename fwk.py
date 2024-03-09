import os
import sys
import subprocess
import re
import asyncio
import typing
import wave
import redis
import openai
import colorama
import sounddevice as sd
import numpy as np
from io import BytesIO
from typing import Any, List, Callable, Tuple
from queue import Queue
from pydub import AudioSegment
from dotenv import load_dotenv

# Third-Parties

load_dotenv()

r = redis.Redis()

openai.api_key = os.environ.get("OPENAI_API_KEY")


# Types

T = typing.TypeVar('T')

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

class SpqdError(Exception):
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
    def __init__(self, name_from: str = "Unknown", name_to: str = "All", data: Any = None) -> None:
        self.name_from = name_from
        self.name_to = name_to
        self.data = data


class Stream:
    def __init__(self, task: Any = None, subs: List[StreamEvent] = None) -> None:
        if subs is None:
            subs = []

        self.coroutine = task
        self.subs = subs

    async def emit(self, event: StreamEvents, data) -> None:
        tasks = [asyncio.create_task(sub.soft(event, data)) for sub in self.subs]
        await asyncio.gather(*tasks)

    async def write(self, value) -> None:
        await self.emit('data', value)

    async def info(self, name_from: str, name_to: str, data: Any) -> None:
        info_bus = StreamBus(name_from, name_to, data)
        await self.emit('info', info_bus)

    async def close(self, value=None) -> None:
        await self.emit('close', value)

    async def error(self, e: BaseException) -> None:
        await self.emit('error', e)

    def task(self, coroutine: Any) -> None:
        self.coroutine = coroutine

    def on(self, event: StreamEvents, callback: typing.Callable) -> None:
        self.subs.append(StreamEvent(event, callback))


class AsstTool:
    def __init__(self, name: str, descr: str, args: dict[str, Any], required: List[str], handler: Callable) -> None:
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


# Functions

def error(message: str) -> SpqdError:
    return SpqdError(message)


def clear_queue(queue: Queue) -> None:
    while not queue.empty():
        queue.get_nowait()


def stdin() -> Stream:
    stdin_stream = Stream()

    # Reading the stdin
    async def read_stdin() -> None:
        try:
            while True:
                # Reading the new chunk
                chunk = sys.stdin.buffer.read(1024)
                if not chunk:
                    await stdin_stream.close()
                    return

                # Streaming the stdin
                await stdin_stream.write(chunk)
        except Exception as e:
            await stdin_stream.error(e)
            await stdin_stream.close()
            raise e

    # Instantly returning the stream
    reading_task = asyncio.create_task(read_stdin())
    stdin_stream.task(reading_task)

    return stdin_stream


def stdout() -> Stream:
    stdout_stream = Stream()

    async def on_data(chunk: bytes) -> None:
        sys.stdout.buffer.write(chunk)

    stdout_stream.on('data', on_data)

    return stdout_stream


def default_input(strict: bool = False) -> Any:
    try:
        device = sd.query_devices(kind='input')
        return device['index']
    except Exception as e:
        if strict:
            raise e

        return None


def default_output(strict: bool = False) -> Any:
    try:
        device = sd.query_devices(kind='output')
        return device['index']
    except Exception as e:
        if strict:
            raise e

        return None


def audio_to_pcm(audio_bytes: bytes, audio_format: str) -> Tuple[bytes, int, int, str]:
    audio_file = BytesIO(audio_bytes)

    # Decoding the audio
    audio_segment = AudioSegment.from_file(audio_file, format=audio_format)

    pcm_bytes = audio_segment.raw_data
    samplerate = audio_segment.frame_rate
    num_channels = audio_segment.channels
    array_type = audio_segment.array_type

    return pcm_bytes, samplerate, num_channels, array_type


def wav_to_pcm(wav_bytes: bytes) -> Tuple[bytes, int, int, str]:
    return audio_to_pcm(wav_bytes, 'wav')


def pcm_to_wav(pcm_bytes: bytes, samplerate: int, num_channels: int) -> bytes:
    wav_io = BytesIO()

    # Encoding the audio
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(samplerate)
        wav_file.writeframes(pcm_bytes)

    wav_io.seek(0)

    return wav_io.getvalue()


def audio_to_wav(audio_bytes: bytes, audio_format: str) -> bytes:
    pcm_bytes, samplerate, num_channels, array_type = audio_to_pcm(audio_bytes, audio_format)
    wav_bytes = pcm_to_wav(pcm_bytes, samplerate, num_channels)

    return wav_bytes


def record_audio(
        input_device: str | int,
        samplerate: int = 16000,
        num_channels: int = 1
) -> Stream:
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

    # Opening the streams
    ffmpeg = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    recording_stream = Stream()

    # Recording the audio
    async def recording_audio() -> None:
        try:
            while True:
                # Reading the new chunk
                chunk = ffmpeg.stdout.read(1024)
                if not chunk:
                    await recording_stream.close()
                    return

                # Streaming the recording
                await recording_stream.write(chunk)
        except Exception as e:
            ffmpeg.kill()
            await recording_stream.error(e)
            await recording_stream.close()

    # Instantly returning the stream
    recording_task = asyncio.create_task(recording_audio())
    recording_stream.task(recording_task)

    return recording_stream


def play_audio(output_device: int | str = 0, samplerate: int = 48000, num_channels: int = 2, array_type: str = 'int32') -> Stream:
    audio_queue = Queue()
    play_queue = Queue()
    playing_stream = Stream()
    stdout_stream = stdout()

    # The play callback
    def callback(outdata, frames, time, status) -> None:
        if status:
            return

        # Filling the output with zeros by default
        outdata.fill(0)

        # If there is no audio - return
        if audio_queue.empty():
            return

        # If there is no play - get it
        if play_queue.empty():
            np_audio = audio_queue.get()
            play_queue.put(np_audio)

        # Trying to get the numpy audio
        np_play = play_queue.get()

        play_audio = np_play[:frames]
        excess_audio = np_play[frames:]

        # Filling the output with the audio
        outdata[:len(play_audio)] = play_audio

        # Putting the excess data back to queue
        if len(excess_audio) > 0:
            play_queue.put(excess_audio)

    # Opening the SD stream
    audio_stream = sd.OutputStream(
        callback=callback,
        device=output_device,
        samplerate=samplerate,
        channels=num_channels,
        dtype=array_type
    )

    # The stream callbacks
    async def on_data(wav_bytes: bytes) -> None:
        nonlocal samplerate, num_channels, audio_queue

        pcm_bytes, pcm_samplerate, pcm_num_channels, pcm_array_type = wav_to_pcm(wav_bytes)

        if pcm_samplerate != samplerate:
            raise error(f"The written samplerate '{pcm_samplerate}' does not match the initial samplerate '{samplerate}'")
        if pcm_num_channels != num_channels:
            raise error(f"The written num channels '{pcm_num_channels}' does not match the initial num channels '{num_channels}'")

        np_bytes = np.frombuffer(pcm_bytes, dtype=array_type)

        channel_length = int(len(np_bytes) / num_channels)
        np_bytes = np_bytes.reshape(channel_length, num_channels)

        audio_queue.put(np_bytes)

    async def on_info(bus: StreamBus) -> None:
        if bus.name_to != 'player':
            return

        match bus.data:
            case 'order:stop':
                clear_queue(audio_queue)
                clear_queue(play_queue)
            case 'order:pause':
                pass

    async def on_close() -> None:
        audio_stream.stop()

    async def playing_process() -> None:
        audio_stream.start()

    # Working with stream
    playing_stream.on('data', on_data)
    playing_stream.on('info', on_info)
    playing_stream.on('close', on_close)

    # Instantly returning the stream
    playing_task = asyncio.create_task(playing_process())
    playing_stream.task(playing_task)

    return playing_stream


# Framework

def gray(message):
    return colorama.Fore.LIGHTBLACK_EX + message + colorama.Style.RESET_ALL


def green(message):
    return colorama.Fore.GREEN + message + colorama.Style.RESET_ALL


def red(message):
    return colorama.Fore.RED + message + colorama.Style.RESET_ALL


def path(*items):
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, *items)


def tts(model: TtsModel, voice: TtsVoice, input_text: str) -> Stream:
    audio_stream = Stream()

    # Fetching the audio
    async def fetch_audio() -> None:
        nonlocal model, voice, input_text, audio_stream

        # Querying the API
        response = openai.audio.speech.create(
            model=model,
            voice=voice,
            input=input_text,
            response_format='opus',
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


def gpt(model: GptModel, conv: Any) -> Stream:
    answer_stream = Stream()

    async def on_close(value: Any) -> None:
        # TODO: kr: Make an ability to stop answering
        pass

    # Fetching the answer
    async def fetch_answer() -> None:
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

    # Instantly returning the stream
    fetching_task = asyncio.create_task(fetch_answer())
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
    async def commit(input_text: str) -> None:
        nonlocal answering_buffer

        asst_delta = AsstDelta(
            type='chunk',
            text=input_text
        )

        await answering_stream.write(asst_delta)
        answering_buffer = ''

    # Analyzes the bot output searching for the quantizable chunks
    async def analyze(input_text: str) -> None:
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
