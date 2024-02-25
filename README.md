# Open Spqd

**Spqd** is an open sound analysis algorithm to creating an advanced voice assistants, call robots, IVR, and many-many other cool stuff. It's like you're having a conversation with your friend—a real person without having a messy Q/A interface and a robot-like talking.

To use a Spqd you need:

1. **macOS 12** or later (the support of other machines come later)
2. **Python 3.11** or later
3. **ffmpeg 6.1.1** or later
4. An **OpenAI** account or a program that will handle the dialogue flow 

To set up a Spqd you have to:

1. Clone this repository into a desired folder:  
`$ cd /path/to/your/folder`  
`$ gh repo clone kennyromanov/open-spqd`
2. Create a python virtual environment:  
`$ cd open-spqd`  
`$ doc/create_venv.sh`
3. Install [requirements](doc/requirements.txt):  
`$ pip3 install -r doc/requirements.txt`
4. Set up an environment variables:  
`$ cp doc/.env.bak .env`  
`$ vim .env`
5. Run it:  
`$ python3 main.py`

You can also add Spqd to your PATH if you like:
```
PATH="/path/to/open-spqd/bin:${PATH}"
```

You succeeded!

---
**Open Spqd** v1.0 (Alpha)  
by Kenny R
