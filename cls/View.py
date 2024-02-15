import asyncio
import time
import numpy as np
import sounddevice as sd
import fwk


class View:
    def __init__(
            self,
            stt_samplerate: int = 16000,
            stt_num_channels: int = 1,
            stt_model: fwk.SttModel = 'whisper-1',
            stt_prompt: str = 'Обычная речь, разделенная запятыми.',
            tts_samplerate: int = 48000,
            tts_num_channels: int = 2,
            tts_model: fwk.TtsModel = 'tts-1',
            tts_voice: fwk.TtsVoice = 'alloy',
            input_device: str | int = 0,
            input_sensitivity: int = 75,
            is_logging: bool = True,
    ) -> None:
        self.stt_samplerate = stt_samplerate
        self.stt_num_channels = stt_num_channels
        self.stt_model = stt_model
        self.stt_prompt = stt_prompt
        self.tts_samplerate = tts_samplerate
        self.tts_num_channels = tts_num_channels
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.input_device = input_device
        self.input_sensitivity = input_sensitivity
        self.is_logging = is_logging

    def _log(self, message: str) -> None:
        if self.is_logging:
            print(f'(v) {message}')

    def tts(self, message: str) -> fwk.Stream:
        audio_stream = fwk.tts(self.tts_model, self.tts_voice, message)
        return audio_stream

    def hear(self) -> fwk.Stream:
        activation_buffer_size = 32 * 1024  # 32 KB
        record_buffer_size = 10 * 1024 * 1024  # 10 MB
        activation_buffer = bytearray()
        record_buffer = bytearray()
        triggered_at = 0
        is_started = False
        is_triggered = False
        i = 0

        record_process = fwk.record_audio(self.input_device, self.stt_samplerate, self.stt_num_channels)
        hearing_stream = fwk.Stream()

        async def to_controller(message: str):
            await hearing_stream.info('v', 'c', message)

        # Commits the record
        async def commit(input_bytes: bytes) -> None:
            nonlocal i

            self._log(f'Commited x{i + 1}')
            await hearing_stream.write(input_bytes)
            record_buffer.clear()

            i += 1

        # Rejects the record
        async def reject(input_bytes: bytes) -> None:
            nonlocal i

            self._log(f'Rejected x{i + 1}')
            record_buffer.clear()

            i += 1

        # Analyzes the record
        async def analyze(chunk_bytes: bytes) -> None:
            nonlocal activation_buffer, record_buffer, triggered_at, is_started, is_triggered

            # Doing some checks
            if not is_started:
                print('==== Chat is Started ====')
            is_started = True

            # If buffers are filled
            if len(activation_buffer) >= activation_buffer_size:
                activation_buffer.clear()
            if len(record_buffer) >= record_buffer_size:
                self._log('Warning: the record buffer has reached its limit')
                record_buffer.clear()

            # Pushing the chunk to the buffers
            activation_buffer.extend(chunk_bytes)
            record_buffer.extend(chunk_bytes)

            calc_volume = fwk.calc_volume(activation_buffer)

            old_is_triggered = is_triggered
            is_triggered = calc_volume > (100 - self.input_sensitivity)

            # Doing some checks
            if is_triggered and not old_is_triggered:
                triggered_at = time.time()
                await to_controller('event:hearing_started')

                self._log(f'Hear you x{i + 1}')
            if not is_triggered and old_is_triggered:
                not_triggered_at = time.time()

                if not_triggered_at - triggered_at > 1:
                    await commit(record_buffer)
                    await to_controller('event:hearing_ended')
                else:
                    await reject(record_buffer)
                    await to_controller('event:hearing_aborted')
            if not is_triggered:
                record_buffer.clear()

        async def hearing_process() -> None:
            try:
                while True:
                    # Reading the stream
                    chunk = record_process.stdout.read(1024)
                    if not chunk:
                        await hearing_stream.close()
                        return

                    # Analyzing the chunk
                    await asyncio.create_task(analyze(chunk))
            except BaseException as e:
                record_process.kill()
                await hearing_stream.error(e)

        hearing_task = asyncio.create_task(hearing_process())
        hearing_stream.task(hearing_task)

        return hearing_stream

    def speak(self) -> fwk.Stream:
        pcm_format = np.int32
        audio_queue = asyncio.Queue()
        excess_queue = asyncio.Queue()
        speaking_stream = fwk.Stream()
        is_speaking = False

        async def to_controller(message: str):
            await speaking_stream.info('v', 'c', message)

        def callback(outdata, frames, time, status):
            nonlocal pcm_format, audio_queue, excess_queue, is_speaking

            old_is_speaking = is_speaking

            if status:
                return

            result = np.zeros((frames, self.tts_num_channels), dtype=pcm_format)

            try:
                # Guessing the queue
                if not excess_queue.empty():
                    final_queue = excess_queue
                else:
                    final_queue = audio_queue

                data = final_queue.get_nowait()
                is_speaking = True

                # Adjusting the chunk size to the frame size
                if data.shape[0] > frames:
                    excess_queue.put_nowait(data[frames:])
                    data = data[:frames]

                for i in range(self.tts_num_channels):
                    result[:data.shape[0], i] = data

                # Marking the task done
                final_queue.task_done()
            except asyncio.QueueEmpty:
                is_speaking = False

            outdata[:] = result

            if is_speaking and not old_is_speaking:
                asyncio.run(to_controller('event:speaking_started'))
            elif not is_speaking and old_is_speaking:
                asyncio.run(to_controller('event:speaking_ended'))

        audio_stream = sd.OutputStream(
            callback=callback,
            samplerate=self.tts_samplerate,
            channels=self.tts_num_channels,
            dtype=pcm_format)

        async def on_data(data) -> None:
            await audio_queue.put(data)

        async def on_info(bus: fwk.StreamBus) -> None:
            if bus.name_to != 'v':
                return
            match bus.data:
                case 'order:stop_speaking':
                    fwk.clear_queue(excess_queue)
                    fwk.clear_queue(audio_queue)

        async def on_close(value) -> None:
            audio_stream.stop()

        async def speaking_process() -> None:
            audio_stream.start()

        speaking_stream.on('data', on_data)
        speaking_stream.on('info', on_info)
        speaking_stream.on('close', on_close)

        speaking_task = asyncio.create_task(speaking_process())
        speaking_stream.task(speaking_task)

        return speaking_stream
