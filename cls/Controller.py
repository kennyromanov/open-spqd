import asyncio
import builtins
import numpy as np
import random
import typing
import fwk
from asyncio import Future
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
        self.control_stream = None

        self.audio_park = {
            'ah': [
                self.audio_gen('А?'),
                self.audio_gen('А?'),
            ],
            'what': [
                self.audio_gen('Что?'),
                self.audio_gen('Что?'),
            ],
            'yes': [
                self.audio_gen('Да-да.'),
                self.audio_gen('Да-да.'),
            ],
            'please': [
                self.audio_gen('Говорите.'),
                self.audio_gen('Говорите.'),
            ],
        }

    def _log(self, message: str) -> None:
        print(f'(c) {message}')

    async def audio_play(self, input_bytes: bytes, format_name: str = 'ogg') -> None:
        bytes_pcm = fwk.to_pcm(input_bytes, format_name)
        array_pcm = np.frombuffer(bytes_pcm, dtype=self.pcm_format)
        await self.speaking_stream.write(array_pcm)

    def audio_tts(self, message: str) -> None:
        tts_stream = self.view.tts(message)
        tts_stream.on('data', self.audio_play)

    def audio_filler(self) -> None:
        random_cat = random.choice(list(self.audio_park.items()))
        random_word = random.choice(random_cat)
        self.audio_play(random_word)

    def audio_gen(self, message: str) -> Future[bytes]:
        future = Future()
        result = bytearray()

        async def on_data(chunk):
            result.extend(chunk)

        async def on_close(value):
            future.set_result(result)

        audio_stream = self.view.tts(message)
        audio_stream.on('data', on_data)
        audio_stream.on('close', on_close)

        return future

    async def start(self) -> None:
        self.hearing_stream = self.view.hear()
        self.speaking_stream = self.view.speak()
        self.control_stream = self.view.control()
        is_speaking = False
        i = 0

        async def to_view(data: typing.Any) -> None:
            await self.control_stream.info('c', 'v', data)

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

            self._log(f'Answering...')
            answering_stream = fwk.asst(transcription)
            answering_stream.on('data', answering_on_data)
            answering_stream.on('error', answering_on_error)
            await answering_stream.coroutine

            i += 1

        async def on_info(bus: fwk.StreamBus) -> None:
            nonlocal is_speaking
            print(bus.data)

            if bus.name_to != 'c':
                return
            print('here')
            match bus.data:
                case 'event:hearing_started':
                    if not is_speaking:
                        return
                    await to_view('order:stop_speaking')
                    self.audio_filler()
                case 'event:speaking_started':
                    is_speaking = True
                case 'event:speaking_ended':
                    is_speaking = False

        async def on_close() -> None:
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
        self.speaking_stream.on('error', on_error)
        self.control_stream.on('error', on_error)

        await self.hearing_stream.coroutine
        await self.speaking_stream.coroutine
        await self.control_stream.coroutine
