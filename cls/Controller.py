import asyncio
import builtins
import numpy as np
import tensorflow as tf
import typing
import random
import time
import colorama
import fwk
from typing import List
from .View import View, HearingPluginData
from .Yamnet import Yamnet


class Controller:
    def __init__(
            self,
            input_device: str | int = 0,
            input_sensitivity: int = 75,
            input_samplerate: int = 16000,
            input_num_channels: int = 1,
    ):
        self.input_device = input_device
        self.input_sensitivity = input_sensitivity
        self.input_samplerate = input_samplerate
        self.input_num_channels = input_num_channels
        self.view: View = View(
            input_device=self.input_device,
            input_sensitivity=self.input_sensitivity,
            stt_samplerate=self.input_samplerate,
            stt_num_channels=self.input_num_channels,
        )
        self.pcm_format = np.int32
        self.hearing_stream = None
        self.speaking_stream = None
        self.yamnet = Yamnet()

    def _log(self, message: str) -> None:
        print(f'(c) {message}')

    def _vad(self, input_bytes: bytes, samplerate: int = 16000) -> float:
        # Convert from pcm to the numpy array
        audio_np = np.frombuffer(input_bytes, dtype=np.int16)
        audio_np = audio_np.astype(np.float32) / 32768.0

        # Resample if the samplerate is different from 16000
        if samplerate != 16000:
            audio_np = tf.signal.resample(audio_np, int(len(audio_np) * 16000 / samplerate))

        # Processing the audio
        analyze = self.yamnet.analyze(audio_np)
        result = self.yamnet.index(analyze)

        return result['Speech'] if result is not None else 0

    def _sv(self, voice_vector: bytes):
        # TODO: Make a speaker verification via the voice vector
        pass

    async def audio_play(self, input_bytes: bytes, format_name: str = 'ogg') -> None:
        bytes_pcm = fwk.to_pcm(input_bytes, format_name)
        array_pcm = np.frombuffer(bytes_pcm, dtype=self.pcm_format)
        await self.speaking_stream.write(array_pcm)

    def audio_tts(self, message: str) -> None:
        tts_stream = self.view.tts(message)
        tts_stream.on('data', self.audio_play)

    async def audio_filler(self, variants: List[str | None] = None) -> None:
        variants = [
            'А?',
            'Что?',
            'Да?..',
            'Да-да?',
            'Простите?',
            'Извините?',
        ]
        if variants is None:
            pass

        variant = random.choice(variants)

        if variant is None:
            return

        self.audio_tts(random.choice(variants))

    async def start(self) -> None:
        is_vad_checking = False
        is_vad_checked = False
        is_vad_passed = False
        is_speaking = False
        is_started = False
        i = 0

        # Checks the audio figuring speech
        async def vad_check(input_bytes: bytes, pass_probability: float = 0.5) -> bool:
            nonlocal is_vad_checking, is_vad_checked

            if is_vad_checking:
                raise fwk.error('Is currently checking')
            if is_vad_checked:
                raise fwk.error('Is already checked')

            is_vad_checking = True
            speech_probability = self._vad(input_bytes)

            is_vad_checked = True
            is_vad_checking = False

            return speech_probability > pass_probability

        # Call on record continuing
        async def on_record_continuing(state: HearingPluginData) -> None:
            nonlocal is_vad_checking, is_vad_checked, is_vad_passed, is_speaking, is_started, i

            if not is_started:
                print('==== Chat is Started ====')

            is_started = True

            if state.is_triggered and state.triggered_duration > 1:
                if not is_speaking or is_vad_checking or is_vad_checked:
                    return

                is_vad_passed = await vad_check(state.record)

                # If speech probability is less than 50%
                if not is_vad_passed:
                    self._log(f'Processed by VAD - Not speaking')
                    return

                # If not - immediately stop speaking and ask the user
                await to_speaking_stream('order:stop_speaking')

                self._log(f'Interrupted')
                await self.audio_filler()

        # Call on record started
        async def on_record_started(state: HearingPluginData) -> None:
            nonlocal is_vad_checked, is_vad_passed, i

            if state.is_triggered and not state.is_prw_triggered:
                is_vad_checked = False
                is_vad_passed = False
                self._log(f'Hear you x{i + 1}')

        # Call on record ended
        async def on_record_ended(state: HearingPluginData) -> None:
            nonlocal is_vad_passed, is_speaking, i

            if not state.is_triggered and state.is_prw_triggered:
                if not is_speaking or is_vad_passed:
                    await state.commit()
                else:
                    if not is_vad_checked:
                        self._log(f'Too short - Not speaking')
                    await state.reject()

                i += 1

        # Open the streams
        self.hearing_stream = self.view.hear([
            on_record_continuing,
            on_record_started,
            on_record_ended,
        ])
        self.speaking_stream = self.view.speak()

        # A message to the speaking stream
        async def to_speaking_stream(data: typing.Any) -> None:
            await self.speaking_stream.info('c', 'v', data)

        # Call on new hearing data
        async def hearing_on_data(data) -> None:
            nonlocal i

            wav_file = fwk.pcm_to_wav(data, self.input_samplerate, self.input_num_channels)
            wav_file.name = 'audio.wav'
            transcription = fwk.stt('whisper-1', wav_file)

            self._log(f'You said: {transcription}')
            if not transcription:
                return

            async def answering_on_data(delta) -> None:
                if delta.type != 'chunk':
                    return

                if not delta.text:
                    return

                self.audio_tts(delta.text)

            async def answering_on_error(e) -> None:
                raise e

            async def answering_process() -> None:
                nonlocal transcription

                answering_stream = fwk.asst(transcription)
                answering_stream.on('data', answering_on_data)
                answering_stream.on('error', answering_on_error)

                await answering_stream.coroutine

            self._log(f'Answering...')
            await asyncio.create_task(answering_process())

        # Call on new speaking message
        async def speaking_on_info(bus: fwk.StreamBus) -> None:
            nonlocal is_speaking

            if bus.name_to != 'c':
                return

            match bus.data:
                case 'event:speaking_started':
                    is_speaking = True
                case 'event:speaking_ended':
                    is_speaking = False

        # Call on close of both
        async def both_on_close(value) -> None:
            pass

        # Call on error of both
        async def both_on_error(e: BaseException) -> None:
            match type(e):
                case builtins.KeyboardInterrupt | asyncio.CancelledError:
                    self._log(f'Interrupted by user')
                case builtins.Warning:
                    self._log(f'Warning: {e}')
                case builtins.Exception:
                    self._log(f'Unexpected Error: {e}')
                case builtins.BaseException | _:
                    self._log(f'Unexpected: {e}')
                    raise e

        self.hearing_stream.on('data', hearing_on_data)
        self.hearing_stream.on('close', both_on_close)
        self.hearing_stream.on('error', both_on_error)
        self.speaking_stream.on('info', speaking_on_info)
        self.speaking_stream.on('error', both_on_error)

        await self.hearing_stream.coroutine
        await self.speaking_stream.coroutine
