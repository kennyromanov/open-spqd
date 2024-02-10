import asyncio
import builtins
import fwk
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

    async def start(self):
        output_stream = await self.view.hear()
        i = 0

        async def on_data(data) -> None:
            nonlocal i

            wav_file = fwk.pcm_to_wav(data, 16000, 1)

            with open(f'tmp/output_{i+1}.wav', 'wb') as output_file:
                output_file.write(wav_file.getvalue())

            self._log(f'Saved x{i+1}')
            i += 1

        async def on_error(e: BaseException) -> None:
            match type(e):
                case builtins.KeyboardInterrupt | asyncio.CancelledError:
                    self._log(f'Interrupted by user')
                case builtins.Warning:
                    self._log(f'Warning: {e.__traceback__}')
                case builtins.Exception:
                    self._log(f'Unexpected Error: {e.__traceback__}')
                case builtins.BaseException | _:
                    self._log(f'Unexpected: {e.__traceback__}')

        output_stream.on('data', on_data)
        output_stream.on('error', on_error)

        await output_stream.task
