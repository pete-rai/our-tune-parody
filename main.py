"""
Our Tune Parody Generator
-------------------------
Generates a parody Radio One "Our Tune" style narration with the iconic
background music of Nino Rota's Romeo and Juliet theme and the cloned
voice of the inimitable Simon Bates.

Our Tune: https://en.wikipedia.org/wiki/Our_Tune
Git Repo: https://github.com/pete-rai/our-tune-parody

MIT License

Ethical Use Notice
------------------
This project is meant for fun, parody, creative or educational use.
Please don't use it to do bad things or to hurt anyone.
Be kind, be sensible and enjoy making your own tragic love stories sound epic.
"""

import torch
import ffmpeg
import soundfile
from chatterbox.tts import ChatterboxTTS

# --- model

MODEL  = ChatterboxTTS.from_pretrained(device = "cuda")  # use "cpu" if needed
CONFIG = {
    "audio_prompt_path": "bates.wav",  # sample of voice to clone
    "exaggeration":       0.75,        # intensity of voice
    "cfg_weight":         0.40,        # pacing of voice
}

# --- config

MUSIC = {
    "start": 14.5,         # seconds into music file to begin
    "gain":  2.5,          # background music gain
    "file":  "music.wav",  # music file path
}

PAUSES = { # all items are in seconds
    "speech": int(MODEL.sr * 0.7),  # pause between speech blocks
    "start":  int(MODEL.sr * 1.0),  # music-only lead-in before speech starts
    "end":    int(MODEL.sr * 2.0),  # tail silence after speech - music fades out here
    "min":    int(MODEL.sr * 0.1),  # minimum inter-block gap
}

# --- transcript

TRANSCRIPT = [ # keep your blocks short - use punctation for pauses, not grammar
    "In the spring of 1912, on the ship they called unsinkable, Rose was a girl trapped in a life of wealth, rules and duty."
    "Then came Jack — a boy with no fortune, just a sketchbook, and a way of looking at her that made her feel truly alive.",
    # long pause
    "They ran through corridors, laughed beneath the stars, and in the back seat of a borrowed car,"
    "left a single handprint that steamed up more than just the window.",
    # long pause
    "But fate was waiting in the cold Atlantic. When the ship went down, there was only one piece of driftwood."
    "Rose lay upon it. Jack did not. She lived because he let her. And he... slipped into the dark waters beneath the stars.",
    # long pause
    "For Rose, life went on. Decades passed. Families grew. But still, in her quietest moments,"
    "she could hear the echo of his voice — promising her she would survive. Even if he could not.",
    # long pause
    "And so, today's Our Tune is for Rose and for Jack. They shared a love too brief to last a lifetime..."
    "And though he passed. Her heart will go on."
]

# --- create a silent waveform of a given size

def silence(waveform, duration):
    return torch.zeros(waveform.size(0), duration)

# --- concatenate waveforms

def concat(waveforms, gap = PAUSES["min"]):
    pause = silence(waveforms[0], gap)
    merge = [waveforms[0]]

    for waveform in waveforms[1:]:
        merge.append(pause)
        merge.append(waveform)

    return torch.cat(merge, dim = 1)

# --- turn text to waveform

def say(text):
    return MODEL.generate(text, **CONFIG)

# --- create whole speech waveform

def speech(transcripts):
    each = [say(text) for text in transcripts]
    all  = concat(each, PAUSES["speech"])
    return all

# --- return the duration of a waveform

def duration(waveform):
    return waveform.size(1) / MODEL.sr

# --- save waveform to a wav file

def save(waveform, output):
    soundfile.write(output, waveform.t().cpu().numpy(), samplerate = MODEL.sr)

# --- create a waveform mixed with background music

def mix(waveform, fadeout, output):
    audio = waveform.cpu().numpy().squeeze().reshape(-1, 1)
    voice = (
        ffmpeg
        .input("pipe:0", format = "f32le", ac = 1, ar = MODEL.sr)
        .audio
    )
    music = (
        ffmpeg
        .input(MUSIC["file"], ss = MUSIC["start"])
        .audio
        .filter("volume", MUSIC["gain"])
        .filter("afade", t = "out", st = fadeout, d = PAUSES["end"] / MODEL.sr)
    )

    mixed = ffmpeg.filter([voice, music], "amix", inputs = 2, duration = "first")
    merge = ffmpeg.output(mixed, output, **{"map_metadata": "-1"}, acodec = "libmp3lame")
    merge.run(input = audio.astype("float32").tobytes(), overwrite_output = True)

# ===== Pipeline =====

if __name__ == "__main__":
    voice = speech(TRANSCRIPT)
    intro = silence(voice, PAUSES["start"])
    outro = silence(voice, PAUSES["end"])
    total = concat([intro, voice, outro])
    fade  = duration(voice) + duration(intro)

    mix(total, fade, "our-tune.mp3")
