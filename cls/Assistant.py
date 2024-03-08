import fwk
from typing import Any
from .SoundMask import SoundMask


class Assistant:
    def __init__(
            self,
            input_stream: fwk.Stream,
            output_stream: fwk.Stream,
            activation_mask: str = None,
            interruption_mask: str = None,
    ) -> None:
        if activation_mask is None:
            activation_mask = ('||.5--<<.5--^^30--VOICE-- ||2--<<1--VOICE--!--')
        if interruption_mask is None:
            interruption_mask = ('||1.5--<<.5--VOICE--')

        self.input_stream = input_stream
        self.output_stream = output_stream
        self.activation_mask = activation_mask
        self.interruption_mask = interruption_mask

    def log(self, message: str) -> None:
        print(f'(a) {message}')

    async def start(self) -> None:
        record_buffer = bytearray()
        commit_buffer = bytearray()
        is_bot_speaking = False
        is_recording = False

        async def sv_callback(wav_bytes: bytes, value: Any) -> bool:
            print('called!')
            return True

        activation_mask = SoundMask(self.activation_mask, {'SV': sv_callback})
        interruption_mask = SoundMask(self.interruption_mask, {'SV': sv_callback})

        async def commit(wav_bytes: bytes) -> None:
            pass

        async def on_data(wav_chunk: bytes) -> None:
            nonlocal record_buffer, is_recording, activation_mask, interruption_mask

            record_buffer.extend(wav_chunk)
            commit_buffer.extend(wav_chunk)

            is_triggered = await activation_mask.test(record_buffer)
            is_interrupted = await interruption_mask.test(record_buffer)

            # If recording started
            if is_triggered and not is_recording:
                self.log('Hear you')
                commit_buffer.clear()
                is_recording = True

            # If recording ended
            if not is_triggered and is_recording:
                self.log('Commited')
                await commit(commit_buffer)
                is_recording = False

            # # If interrupted
            # if is_interrupted and is_recording:
            #     self.log('Interrupted')
            #     is_recording = False

        async def on_info(message: str) -> None:
            pass

        async def on_error(e: Exception) -> None:
            self.log(f'Unexpected error: {e}')

        self.input_stream.on('data', on_data)
        self.input_stream.on('info', on_info)
        self.input_stream.on('error', on_error)

        print('==== Session started ====')
