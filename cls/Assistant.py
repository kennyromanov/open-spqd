import fwk
from typing import Any
from .SoundMask import SoundMask


class Assistant:
    def __init__(
            self,
            sound_mask: str = None,
    ) -> None:
        if sound_mask is None:
            sound_mask = ('>>1---^^80----- ---<<2--vv70--! '
                          '||.5--VOICE---- --------------- '
                          '||.5--::1------ --------------- ')

        self.sound_mask = sound_mask

    async def start(self):
        async def callback(wav_bytes: bytes, value: Any) -> bool:
            print('called!')
            return True

        sound_mask = SoundMask('---<<2--vv70--!', {
            1: callback
        })

        with open(fwk.path('tmp', 'Test-1.wav'), 'rb') as file:
            silent_sound = file.read()
        with open(fwk.path('tmp', 'Test-2.wav'), 'rb') as file:
            loud_sound = file.read()

        print(await sound_mask.test(silent_sound))
        print(await sound_mask.test(loud_sound))
