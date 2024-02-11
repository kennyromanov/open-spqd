import asyncio
import builtins
import time

import fwk
from asyncio import Future
from .View import View


class Controller:
    def __init__(
            self,
            input_sensitivity: int = 75,
            input_device: str | int = 0,
    ):
        self.view: View = View(
            input_sensitivity=input_sensitivity,
            input_device=input_device
        )

    def _log(self, message: str) -> None:
        print(f'(c) {message}')

    async def start(self) -> None:
        output_stream = self.view.hear()
        i = 0

        async def on_data(data) -> None:
            nonlocal i

            wav_file = fwk.pcm_to_wav(data, 16000, 1)

            template = f'tmp/output_{i+1}.wav'
            with open(template, 'wb') as output_file:
                output_file.write(wav_file.getvalue())

            self._log(f'Working on... x{i+1}')
            transcription = fwk.stt('whisper-1', template)

            self._log(f'You said: {transcription}')
            await self.view.say(transcription)

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

        output_stream.on('data', on_data)
        output_stream.on('close', on_close)
        output_stream.on('error', on_error)

        await output_stream.task_coroutine
