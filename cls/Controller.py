import asyncio
import builtins
import numpy as np
import typing
import random
import colorama
import fwk
from typing import List
from .View import View


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

    def _log(self, message: str) -> None:
        print(f'(c) {message}')

    async def audio_play(self, input_bytes: bytes, format_name: str = 'ogg') -> None:
        bytes_pcm = fwk.to_pcm(input_bytes, format_name)
        array_pcm = np.frombuffer(bytes_pcm, dtype=self.pcm_format)
        await self.speaking_stream.write(array_pcm)

    def audio_tts(self, message: str) -> None:
        tts_stream = self.view.tts(message)
        tts_stream.on('data', self.audio_play)

    async def audio_filler(self, variants: List[str | None] = None) -> None:
        if variants is None:
            variants = [
                'А?',
                'Да?..',
                'Что-что?',
                'Простите?',
                'Простите, Вы что-то сказали?',
                'Извините, что?',
                'Да-да? Чем я могу помочь?',
            ]

        variant = random.choice(variants)

        if variant is None:
            return

        self.audio_tts(random.choice(variants))

    async def start(self) -> None:
        self.hearing_stream = self.view.hear()
        self.speaking_stream = self.view.speak()
        is_speaking = False
        i = 0

        async def to_hearing(data: typing.Any) -> None:
            await self.hearing_stream.info('c', 'v', data)

        async def to_speaking(data: typing.Any) -> None:
            await self.speaking_stream.info('c', 'v', data)

        async def on_data(data) -> None:
            nonlocal i

            wav_file = fwk.pcm_to_wav(data, self.input_samplerate, self.input_num_channels)

            template = f'tmp/output_{i+1}.wav'
            with open(template, 'wb') as output_file:
                output_file.write(wav_file.getvalue())

            transcription = fwk.stt('whisper-1', template)

            self._log(f'You said: {transcription}')
            if not transcription:
                i += 1
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

            i += 1

        async def on_info(bus: fwk.StreamBus) -> None:
            nonlocal is_speaking
            # self._log(f'{colorama.Fore.RED}dbg:{bus.data}{colorama.Style.RESET_ALL}')

            if bus.name_to != 'c':
                return

            match bus.data:
                case 'event:hearing_started':
                    # A regular hearing - not interrupting
                    if not is_speaking:
                        return

                    # Immediately stop speaking and ask the user
                    await to_speaking('order:stop_speaking')
                    self._log(f'Interrupted')
                case 'event:hearing_aborted':
                    self._log(f'Filling...')
                    await self.audio_filler()
                case 'event:speaking_started':
                    is_speaking = True
                case 'event:speaking_ended':
                    is_speaking = False

        async def on_close(value) -> None:
            pass

        async def on_error(e: BaseException) -> None:
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

        self.hearing_stream.on('data', on_data)
        self.hearing_stream.on('info', on_info)
        self.hearing_stream.on('close', on_close)
        self.hearing_stream.on('error', on_error)
        self.speaking_stream.on('info', on_info)
        self.speaking_stream.on('error', on_error)

        await self.hearing_stream.coroutine
        await self.speaking_stream.coroutine
