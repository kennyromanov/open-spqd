import asyncio
import time
import fwk


class View:
    def __init__(
            self,
            stt_model: fwk.SttModel = 'whisper-1',
            stt_prompt: str = 'Обычная речь, разделенная запятыми.',
            tts_model: fwk.TtsModel = 'tts-1',
            tts_voice: fwk.TtsVoice = 'alloy',
            input_sensitivity: int = 75,
            input_device: str | int = 0,
            is_logging: bool = True,
    ):
        self.stt_model = stt_model
        self.stt_prompt = stt_prompt
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.input_sensitivity = input_sensitivity
        self.input_device = input_device
        self.is_logging = is_logging

    def _log(self, message: str) -> None:
        if self.is_logging:
            print(f'(v) {message}')

    async def say(self, message: str) -> None:
        queue = asyncio.Queue()

        async def on_data(chunk):
            nonlocal queue
            await queue.put(chunk)

        async def on_close(value):
            nonlocal queue
            await queue.put(value)

        async def play_audio():
            nonlocal queue

            while True:
                chunk = await queue.get()

                if chunk is None:
                    queue.task_done()
                    break

                await fwk.play_opus(chunk, 'ogg')
                time.sleep(0.1)

            queue.task_done()

        audio_stream = fwk.tts(self.tts_model, self.tts_voice, message)
        audio_stream.on('data', on_data)
        audio_stream.on('close', on_close)

        fetch_task = audio_stream.task
        play_task = asyncio.create_task(play_audio())

        await fetch_task
        await play_task

    async def hear(self) -> fwk.Stream:
        frame_rate = 16000
        num_channels = 1
        chunk_size = 1024
        activation_buffer_size = 32 * 1024  # 32 KB
        record_buffer_size = 10 * 1024 * 1024  # 10 MB
        activation_buffer = bytearray()
        record_buffer = bytearray()
        is_started = False
        is_triggered = False
        i = 0

        record_process = fwk.record_audio(self.input_device, frame_rate, num_channels, chunk_size)
        output_stream = fwk.Stream()

        # Commits the record
        async def commit(input_bytes: bytes) -> None:
            nonlocal i

            await output_stream.write(input_bytes)
            record_buffer.clear()

            self._log(f'Commited x{i+1}')
            i += 1

        # Analyzes the record
        async def analyze(chunk_bytes: bytes) -> None:
            nonlocal activation_buffer, record_buffer, is_started, is_triggered

            # Doing some checks
            if not is_started:
                print('==== Chat is Started ====')
            is_started = True

            # If buffers are filled
            if len(activation_buffer) >= activation_buffer_size:
                activation_buffer.clear()
            if len(record_buffer) >= record_buffer_size:
                record_buffer.clear()

            # Pushing the chunk to the buffers
            activation_buffer.extend(chunk_bytes)
            record_buffer.extend(chunk_bytes)

            calc_volume = fwk.calc_volume(activation_buffer)

            old_is_triggered = is_triggered
            is_triggered = calc_volume > (100 - self.input_sensitivity)

            # Doing some checks
            if is_triggered and not old_is_triggered:
                self._log(f'Hear you x{i+1}')
            if not is_triggered and old_is_triggered:
                await commit(record_buffer)
            if not is_triggered:
                record_buffer.clear()

        async def analyzing_process() -> None:
            try:
                while True:
                    # Reading the stream
                    chunk = record_process.stdout.read(1024)
                    if not chunk:
                        break

                    # Analyzing the chunk
                    await asyncio.create_task(analyze(chunk))
            except BaseException as e:
                record_process.kill()
                await output_stream.error(e)

        analyzing_task = asyncio.create_task(analyzing_process())
        output_stream.set(analyzing_task)

        return output_stream
