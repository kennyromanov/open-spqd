import asyncio
import builtins
import numpy as np
import fwk
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

    def _log(self, message: str) -> None:
        print(f'(c) {message}')

    async def start(self) -> None:
        pcm_format = np.int32
        speaking_stream = self.view.speak()
        hearing_stream = self.view.hear()
        i = 0

        async def on_data(data) -> None:
            nonlocal i

            wav_file = fwk.pcm_to_wav(data, self.input_samplerate, self.input_num_channels)

            template = f'tmp/output_{i+1}.wav'
            with open(template, 'wb') as output_file:
                output_file.write(wav_file.getvalue())

            self._log(f'Working on... x{i+1}')
            transcription = fwk.stt('whisper-1', template)

            self._log(f'You said: {transcription}')
            if not transcription:
                i += 1
                return

            tts_stream = self.view.tts(transcription)

            async def tts_on_data(chunk) -> None:
                bytes_pcm = fwk.to_pcm(chunk, 'ogg')
                array_pcm = np.frombuffer(bytes_pcm, dtype=pcm_format)
                await speaking_stream.write(array_pcm)

            tts_stream.on('data', tts_on_data)

            i += 1

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
                    self._log(f'Unexpected: {e}')\

        hearing_stream.on('data', on_data)
        hearing_stream.on('close', on_close)
        hearing_stream.on('error', on_error)

        await hearing_stream.task_coroutine
