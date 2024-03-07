import math
import asyncio
import numpy as np
import tensorflow as tf
import fwk
from io import BytesIO
from typing import List, Callable, Awaitable, Any
from pydub import AudioSegment
from .Yamnet import Yamnet


# Third-parties

yamnet = Yamnet()


# Types & Classes

OperatorCallback = Callable[[bytes, Any], Awaitable[bool]]


class Instruction:
    def __init__(self, name: str, mod: str = '', value: str = '') -> None:
        self.name = name.upper()
        self.mod = mod.upper()
        self.value = value.upper()


Pipeline = List[Instruction]


# Constants

BYTE_FALSE = b'\x00'

BYTE_TRUE = b'\x01'


# Base Functions

def calc_duration(wav_bytes: bytes) -> float:
    wav_file = BytesIO(wav_bytes)
    audio_segment = AudioSegment.from_file(wav_file, format='wav')
    return len(audio_segment) / 1000


def calc_volume(wav_bytes: bytes) -> float:
    pcm_bytes, samplerate, num_channels, array_type = fwk.wav_to_pcm(wav_bytes)

    samples = np.frombuffer(pcm_bytes, dtype=array_type)
    calc_sum = np.sum(samples.astype(np.int64) ** 2)
    sample_count = len(samples)

    if sample_count == 0:
        return 0

    rms = math.sqrt(calc_sum / sample_count)
    db = 20 * math.log10((rms + 1e-9) / 32768)

    min_db = -60.0
    max_db = 20.0
    result = (db - min_db) / (max_db - min_db) * 100

    # Normalizing the value from 0 to 100
    result = 100 if result > 100 else result
    result = 0 if result < 0 else result

    return result


def detect_voice(wav_bytes: bytes) -> float:
    pcm_bytes, samplerate, num_channels, array_type = fwk.wav_to_pcm(wav_bytes)

    # Convert from pcm to the numpy array
    audio_np = np.frombuffer(pcm_bytes, dtype=array_type)
    audio_np = audio_np.astype(np.float32) / 32768.0

    # Resample if the samplerate is different from 16000
    if samplerate != 16000:
        audio_np = tf.signal.resample(audio_np, int(len(audio_np) * 16000 / samplerate))

    # Processing the audio
    analyze = yamnet.analyze(audio_np)
    result = yamnet.index(analyze)

    return result['Speech'] if result is not None else 0


def trim_audio(wav_bytes: bytes, sec_start: float, sec_end: float) -> fwk:
    if sec_end < sec_start:
        raise fwk.error('The seconds end param cannot be lower than the seconds start')

    input_file = BytesIO(wav_bytes)
    output_file = BytesIO()

    audio_segment = AudioSegment.from_file(input_file, format='wav')

    trimmed_audio = audio_segment[1000 * sec_start:1000 * sec_end]
    trimmed_audio.export(output_file, format='wav')

    return output_file.getvalue()


# Operator Handlers

async def op_not(wav_bytes: bytes, value: Any) -> bytes:
    return BYTE_FALSE if wav_bytes else BYTE_TRUE


async def op_first_sec(wav_bytes: bytes, value: float) -> bytes:
    return trim_audio(wav_bytes, 0, value)


async def op_last_sec(wav_bytes: bytes, value: float) -> bytes:
    duration = calc_duration(wav_bytes)
    return trim_audio(wav_bytes, duration - value, duration)


async def op_delay_sec(wav_bytes: bytes, value: float) -> bytes:
    duration = calc_duration(wav_bytes)
    return trim_audio(wav_bytes, value, duration)


async def op_volume_above(wav_bytes: bytes, reference: float) -> bytes:
    return wav_bytes if calc_volume(wav_bytes) >= reference else BYTE_FALSE


async def op_volume_below(wav_bytes: bytes, reference: float) -> bytes:
    return wav_bytes if calc_volume(wav_bytes) < reference else BYTE_FALSE


async def op_voice_above(wav_bytes: bytes, reference: float) -> bytes:
    return wav_bytes if 100 * detect_voice(wav_bytes) >= reference else BYTE_FALSE


async def op_ext_call(wav_bytes: bytes, callback: OperatorCallback) -> bytes:
    return wav_bytes if await callback(wav_bytes, None) else BYTE_FALSE


# The SoundMask class

class SoundMask:
    def __init__(self, expression: str, callbacks: dict[str | int, OperatorCallback] = None) -> None:
        if not callbacks:
            callbacks = {}

        self.handlers = {
            'NOT': op_not,
            'FIRST SEC': op_first_sec,
            'LAST SEC': op_last_sec,
            'DELAY SEC': op_delay_sec,
            'VOLUME ABOVE': op_volume_above,
            'VOLUME BELOW': op_volume_below,
            'VOICE ABOVE': op_voice_above,
            'EXT CALL': op_ext_call,
        }
        self.expression = expression
        self.callbacks = callbacks

    def to_instructions(self, expression: str) -> List[str]:
        instructions: List[str] = []
        buffer = ''

        def commit() -> None:
            nonlocal instructions, buffer

            parsed = buffer
            display = parsed[:2].upper()
            value = parsed[2:]
            instruction = ''

            # Cutoff the non-instruction
            if parsed == '':
                buffer = ''
                return

            # Match the independent operators
            match parsed:
                case ' ':
                    instruction = 'AND'
                case 'NOT' | '!':
                    instruction = 'NOT'
                case 'VOICE':
                    instruction = 'VOICE ABOVE 50'

            # Match the value operators
            match display:
                case '>>':
                    value = float(value)
                    instruction = f'FIRST SEC {value}'
                case '<<':
                    value = float(value)
                    instruction = f'LAST SEC {value}'
                case '||':
                    value = float(value)
                    instruction = f'DELAY SEC {value}'
                case '^^':
                    value = int(value)
                    instruction = f'VOLUME ABOVE {value}'
                case 'VV':
                    value = int(value)
                    instruction = f'VOLUME BELOW {value}'
                case '::':
                    value = int(value)
                    instruction = f'EXT CALL {value}'

            if not instruction:
                raise fwk.error(f"SoundMask: Unknown operator '{parsed}'")

            instructions.append(instruction)
            buffer = ''

        for char in expression:
            if char == '-' or char == ' ' or buffer == ' ':
                commit()

                if char == '-':
                    continue

            buffer += char

        commit()

        return instructions

    def to_pipelines(self, instructions: List[str]) -> List[Pipeline]:
        pipelines: List[Pipeline] = []
        result: List[Pipeline] = []
        buffer: Pipeline = []

        for instruction in instructions:
            parts = instruction.split(' ')

            if len(parts) == 1:
                instruction = Instruction(name=parts[0])
            else:
                instruction = Instruction(
                    name=parts[0],
                    mod=parts[1],
                    value=parts[2],
                )

            match instruction.name:
                case 'AND':
                    pipelines.append(buffer)
                    buffer = []
                    pass
                case _:
                    buffer.append(instruction)

            pipelines.append(buffer)

        for pipeline in pipelines:
            if len(pipeline) > 0:
                result.append(pipeline)

        return result

    async def test(self, wav_bytes: bytes) -> bool:
        instructions = self.to_instructions(self.expression)
        pipelines = self.to_pipelines(instructions)

        async def pipline_processor(input_wav: bytes, pipline: Pipeline) -> bool:
            # Iterating the pipeline
            for instruction in pipline:
                inst_name = instruction.name.upper()+' '+instruction.mod.upper()
                inst_name = inst_name.strip()
                inst_mod = instruction.mod.upper()
                inst_value = instruction.value

                # Matching the value type
                match inst_mod:
                    case 'SEC' | 'ABOVE' | 'BELOW':
                        inst_value = float(inst_value)
                    case 'CALL':
                        callback = self.callbacks[inst_value]

                        if not callback:
                            raise fwk.error(f'SoundMask: No callback named {inst_value}')

                        inst_value = callback

                # Calculating the value
                input_wav = await self.handlers[inst_name](input_wav, inst_value)

                if input_wav in (BYTE_TRUE, BYTE_FALSE):
                    break

            return input_wav != BYTE_FALSE

        tasks = [pipline_processor(wav_bytes, x) for x in pipelines]
        result = all(await asyncio.gather(*tasks))

        return result
