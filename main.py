"""
Open SpeakQ Service 1
by Kenny Romanov
"""

import asyncio

# from argparse import ArgumentParser
#
# parser = ArgumentParser()
# parser.add_argument("--voice", default="-")
# parser.add_argument("--sensitivity", default="-")
# parser.add_argument("--input", default="-")
# args = parser.parse_args()

# import tst.TTS as Test
import tst.Record as Test

asyncio.run(Test.main())
