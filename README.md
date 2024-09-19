# Open Spqd

**Spqd** is an open-source audio analysis algorithm for creating advanced voice assistants, call robots, IVR, real-world agents and many, many other cool stuff. It's like talking to your real—life friend without having a messy Q/A interface and robotic speech.

To use Spqd you need:

1. **macOS 10.7**, **Linux PulseAudio 1.0** or later (the support of other machines come later)
2. **Python 3.10** or later
3. **ffmpeg 2.2** or later
4. An **OpenAI** account or a script that will handle [the Assistant](cls/Assistant.py) 

To set up Spqd you have to:

1. Clone this repository into a desired folder:  
`$ cd /path/to/your/folder`  
`$ gh repo clone kennyromanov/open-spqd`
2. Create a python virtual environment:  
`$ cd open-spqd`  
`$ bin/create-venv`
3. Install [requirements](doc/requirements.txt):  
`$ pip3 install -r requirements.txt`
4. Set up environment variables:  
`$ cp .env.example .env`  
`$ vim .env`
5. Run:  
`$ bin/spqd`

You can also add Spqd to your PATH if you like:
```
PATH="/path/to/open-spqd/bin:${PATH}"
```

You did it!

---
**Open Spqd** v1.2 (Alpha)  
by Kenny R
