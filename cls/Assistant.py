import fwk
from typing import Any, Callable, Awaitable
from .SoundMask import SoundMask, calc_duration


class AssistantState:
    def __init__(self, message: str) -> None:
        self.message = message


AssistantHandler = Callable[[AssistantState], fwk.Stream]


def base_handler(state: AssistantState) -> fwk.Stream:
    answering_stream = fwk.asst(state.message)
    output_stream = fwk.Stream()
    bot_message = ''

    print(f'(?) "{state.message}"')
    print(f'(>) ', end='')

    async def on_data(delta: fwk.AsstDelta) -> None:
        nonlocal output_stream, bot_message

        if delta.type == 'stop':
            print()
            await output_stream.close()
            return

        bot_message += delta.text
        print(delta.text, end='')

        await output_stream.write(delta.text)

    answering_stream.on('data', on_data)
    output_stream.task(answering_stream.coroutine)

    return output_stream


class Assistant:
    def __init__(
            self,
            voice: fwk.TtsVoice,
            sensitivity: int,
            input_stream: fwk.Stream,
            output_stream: fwk.Stream,
            assistant_handler: AssistantHandler = None,
            activation_mask: str = None,
            deactivation_mask: str = None,
            speaking_mask: str = None,
    ) -> None:
        if assistant_handler is None:
            assistant_handler = base_handler
        if activation_mask is None:
            activation_mask = f'<<.5--^^{100-sensitivity}--'
        if deactivation_mask is None:
            deactivation_mask = f'<<1--vv{100-sensitivity}--'
        if speaking_mask is None:
            speaking_mask = '<<2--VOICE--'

        self.voice = voice
        self.sensitivity = sensitivity
        self.input_stream = input_stream
        self.output_stream = output_stream
        self.assistant_handler = assistant_handler
        self.activation_mask = activation_mask
        self.deactivation_mask = deactivation_mask
        self.speaking_mask = speaking_mask

    def log(self, message: str) -> None:
        print(f'(c) {message}')

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

        # The messenger function
        async def player(message: str) -> None:
            await self.output_stream.info('assistant', 'player', message)

        async def commit(pcm_bytes: bytes) -> None:
            wav_bytes = fwk.pcm_to_wav(pcm_bytes, 16000, 1)

            state = AssistantState(
                message=fwk.stt('whisper-1', wav_bytes, 'Обычная речь, разделенная запятыми.')
            )
            answering_stream = self.assistant_handler(state)

            async def answering_on_data(text_chunk: str):
                tts_stream = fwk.tts('tts-1', self.voice, text_chunk)

                async def tts_on_data(ogg_chunk: bytes) -> None:
                    tts_bytes = fwk.audio_to_wav(ogg_chunk, 'ogg')
                    await self.output_stream.write(tts_bytes)

                tts_stream.on('data', tts_on_data)
                tts_stream.on('info', on_info)
                tts_stream.on('error', on_error)

                await tts_stream.coroutine

            answering_stream.on('data', answering_on_data)
            answering_stream.on('info', on_info)
            answering_stream.on('error', on_error)

            await answering_stream.coroutine

        async def on_data(pcm_chunk: bytes) -> None:
            nonlocal \
                record_buffer, commit_buffer, \
                activation_mask, deactivation_mask, speaking_mask, \
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

                # If interrupted - immediately stop speaking
                if is_bot_speaking:
                    self.log('Interrupted')
                    await player('order:stop')

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

        async def on_info(bus: fwk.StreamBus) -> None:
            nonlocal is_bot_speaking

            if bus.name_to not in ('assistant', ''):
                return

            match bus.name_from:
                case 'player':
                    match bus.data:
                        case 'event:play':
                            is_bot_speaking = True
                        case 'event:stop':
                            is_bot_speaking = False

        async def on_error(e: Exception) -> None:
            self.log(f'Unexpected error: {e}\n')
            raise e

        self.input_stream.on('data', on_data)
        self.input_stream.on('info', on_info)
        self.output_stream.on('info', on_info)
        self.input_stream.on('error', on_error)

        print('==== Session started ====')
