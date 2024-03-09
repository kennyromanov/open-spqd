import fwk
from typing import Any
from .SoundMask import SoundMask, calc_duration


class Assistant:
    def __init__(
            self,
            input_stream: fwk.Stream,
            output_stream: fwk.Stream,
            activation_mask: str = None,
            deactivation_mask: str = None,
            speaking_mask: str = None,
            interruption_mask: str = None,
    ) -> None:
        if activation_mask is None:
            activation_mask = ('<<.5--^^30--')
        if deactivation_mask is None:
            deactivation_mask = ('<<1--vv30--')
        if speaking_mask is None:
            speaking_mask = ('<<2--VOICE--')
        if interruption_mask is None:
            interruption_mask = ('<<.3--VOICE--')

        self.input_stream = input_stream
        self.output_stream = output_stream
        self.activation_mask = activation_mask
        self.deactivation_mask = deactivation_mask
        self.speaking_mask = speaking_mask
        self.interruption_mask = interruption_mask

    def log(self, message: str) -> None:
        print(f'(a) {message}')

    async def start(self) -> None:
        record_buffer = bytearray()
        commit_buffer = bytearray()
        next_nn_time = 0
        next_start_time = 0
        is_recording = False
        is_proved = False
        is_bot_speaking = False

        async def sv_callback(wav_bytes: bytes, value: Any) -> bool:
            print('called!')
            return True

        activation_mask = SoundMask(self.activation_mask, {'SV': sv_callback})
        deactivation_mask = SoundMask(self.deactivation_mask, {'SV': sv_callback})
        speaking_mask = SoundMask(self.speaking_mask, {'SV': sv_callback})
        interruption_mask = SoundMask(self.interruption_mask, {'SV': sv_callback})

        async def commit(wav_bytes: bytes) -> None:
            tts_stream = fwk.tts('tts-1', 'nova', 'Тщательно проверяем любые веганские товары: еду, косметику, товары гигиены и бытовую химию. Помогаем веганизировать образ жизни взрослых, детей и питомцев. Постоянно учимся новому и делимся этим с вами.')

            async def tts_on_data(ogg_chunk: bytes) -> None:
                tts_bytes = fwk.audio_to_wav(ogg_chunk, 'ogg')
                await self.output_stream.write(tts_bytes)

            tts_stream.on('data', tts_on_data)
            tts_stream.on('info', on_info)
            tts_stream.on('error', on_error)

            await tts_stream.coroutine

        async def on_data(pcm_chunk: bytes) -> None:
            nonlocal \
                record_buffer, commit_buffer, \
                activation_mask, deactivation_mask, speaking_mask, interruption_mask, \
                next_nn_time, next_start_time, is_recording, is_proved, is_bot_speaking

            record_buffer.extend(pcm_chunk)
            commit_buffer.extend(pcm_chunk)

            wav_recording = fwk.pcm_to_wav(record_buffer, 16000, 1)
            recording_duration = calc_duration(wav_recording)
            do_record = recording_duration >= next_start_time
            do_prove_speaking = is_recording and recording_duration >= next_nn_time

            is_activated = await activation_mask.test(wav_recording) if do_record else False
            is_deactivated = await deactivation_mask.test(wav_recording) if is_recording else False
            is_speaking = await speaking_mask.test(wav_recording) if do_prove_speaking else True
            is_interrupted = await interruption_mask.test(wav_recording) if is_bot_speaking else False

            # If recording started
            if is_activated and not is_recording:
                self.log('Hear you')
                commit_buffer.clear()

                is_recording = True
                is_proved = False
                do_prove_speaking = False
                next_nn_time = recording_duration + 1

            # If prove speaking
            if do_prove_speaking and is_speaking:
                if is_proved:
                    self.log('Checking you...')
                if not is_proved:
                    self.log('Checked you... OK')

                is_proved = True
                next_nn_time = recording_duration + 3

            # If recording ended
            if (is_deactivated and is_recording) or (not is_speaking and is_recording):
                if is_proved:
                    self.log(fwk.green('Commited'))
                    await commit(commit_buffer)
                if not is_proved:
                    self.log(fwk.red('Rejected - speech not proved'))

                is_recording = False
                next_start_time = recording_duration + 1

            # # If interrupted
            # if is_interrupted and is_recording:
            #     self.log('Interrupted')
            #     is_recording = False

        async def on_info(bus: fwk.StreamBus) -> None:
            pass

        async def on_error(e: Exception) -> None:
            self.log(f'Unexpected error: {e}\n')
            raise e

        self.input_stream.on('data', on_data)
        self.input_stream.on('info', on_info)
        self.input_stream.on('error', on_error)

        print('==== Session started ====')
