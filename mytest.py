# coding: utf-8
import argparse
import logging
import os
import pathlib
import time
import tempfile
import platform
import webbrowser
import sys
print(f"default encoding is {sys.getdefaultencoding()},file system encoding is {sys.getfilesystemencoding()}")
print(f"You are using Python version {platform.python_version()}")
if(sys.version_info[0]<3 or sys.version_info[1]<7):
    print("The Python version is too low and may cause problems")

if platform.system().lower() == 'windows':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import langid
langid.set_languages(['en', 'zh', 'ja'])

import nltk
nltk.data.path = nltk.data.path + [os.path.join(os.getcwd(), "nltk_data")]

import torch
import torchaudio
import random

import numpy as np

from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)
from data.collation import get_text_token_collater
from models.vallex import VALLE
from utils.g2p import PhonemeBpeTokenizer
from descriptions import *
from macros import *
from examples import *

import gradio as gr
import whisper
from vocos import Vocos
import multiprocessing

thread_count = multiprocessing.cpu_count()

print("Use",thread_count,"cpu cores for computing")

torch.set_num_threads(thread_count)
torch.set_num_interop_threads(thread_count)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
text_collater = get_text_token_collater()

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)
if torch.backends.mps.is_available():
    device = torch.device("mps")
# VALL-E-X model
if not os.path.exists("./checkpoints/"): os.mkdir("./checkpoints/")
if not os.path.exists(os.path.join("./checkpoints/", "vallex-checkpoint.pt")):
    import wget
    try:
        logging.info("Downloading model from https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt ...")
        # download from https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt to ./checkpoints/vallex-checkpoint.pt
        wget.download("https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt",
                      out="./checkpoints/vallex-checkpoint.pt", bar=wget.bar_adaptive)
    except Exception as e:
        logging.info(e)
        raise Exception(
            "\n Model weights download failed, please go to 'https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt'"
            "\n manually download model weights and put it to {} .".format(os.getcwd() + "\checkpoints"))

model = VALLE(
        N_DIM,
        NUM_HEAD,
        NUM_LAYERS,
        norm_first=True,
        add_prenet=False,
        prefix_mode=PREFIX_MODE,
        share_embedding=True,
        nar_scale_factor=1.0,
        prepend_bos=True,
        num_quantizers=NUM_QUANTIZERS,
    )
checkpoint = torch.load("./checkpoints/vallex-checkpoint.pt", map_location='cpu')
missing_keys, unexpected_keys = model.load_state_dict(
    checkpoint["model"], strict=True
)
assert not missing_keys
model.eval()

# Encodec model
audio_tokenizer = AudioTokenizer(device)

# Vocos decoder
vocos = Vocos.from_pretrained('charactr/vocos-encodec-24khz').to(device)

# ASR
if not os.path.exists("./whisper/"): os.mkdir("./whisper/")
try:
    whisper_model = whisper.load_model("medium",download_root=os.path.join(os.getcwd(), "whisper")).cpu()
except Exception as e:
    logging.info(e)
    raise Exception(
        "\n Whisper download failed or damaged, please go to "
        "'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt'"
        "\n manually download model and put it to {} .".format(os.getcwd() + "\whisper"))

# Voice Presets
preset_list = os.walk("./presets/").__next__()[2]
preset_list = [preset[:-4] for preset in preset_list if preset.endswith(".npz")]

def clear_prompts():
    try:
        path = tempfile.gettempdir()
        for eachfile in os.listdir(path):
            filename = os.path.join(path, eachfile)
            if os.path.isfile(filename) and filename.endswith(".npz"):
                lastmodifytime = os.stat(filename).st_mtime
                endfiletime = time.time() - 60
                if endfiletime > lastmodifytime:
                    os.remove(filename)
    except:
        return

def transcribe_one(model, audio_path):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(temperature=1.0, best_of=5, fp16=False if device == torch.device("cpu") else True, sample_len=150)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)

    text_pr = result.text
    if text_pr.strip(" ")[-1] not in "?!.,。，？！。、":
        text_pr += "."
    return lang, text_pr

def make_npz_prompt(name, uploaded_audio, recorded_audio, transcript_content):
    global model, text_collater, text_tokenizer, audio_tokenizer
    clear_prompts()
    audio_prompt = uploaded_audio if uploaded_audio is not None else recorded_audio
    sr, wav_pr = audio_prompt
    if not isinstance(wav_pr, torch.FloatTensor):
        wav_pr = torch.FloatTensor(wav_pr)
    if wav_pr.abs().max() > 1:
        wav_pr /= wav_pr.abs().max()
    if wav_pr.size(-1) == 2:
        wav_pr = wav_pr[:, 0]
    if wav_pr.ndim == 1:
        wav_pr = wav_pr.unsqueeze(0)
    assert wav_pr.ndim and wav_pr.size(0) == 1

    if transcript_content == "":
        text_pr, lang_pr = make_prompt(name, wav_pr, sr, save=False)
    else:
        lang_pr = langid.classify(str(transcript_content))[0]
        lang_token = lang2token[lang_pr]
        text_pr = f"{lang_token}{str(transcript_content)}{lang_token}"
    # tokenize audio
    encoded_frames = tokenize_audio(audio_tokenizer, (wav_pr, sr))
    audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu().numpy()

    # tokenize text
    phonemes, _ = text_tokenizer.tokenize(text=f"{text_pr}".strip())
    text_tokens, enroll_x_lens = text_collater(
        [
            phonemes
        ]
    )

    message = f"Detected language: {lang_pr}\n Detected text {text_pr}\n"

    # save as npz file
    np.savez(os.path.join(tempfile.gettempdir(), f"{name}.npz"),
             audio_tokens=audio_tokens, text_tokens=text_tokens, lang_code=lang2code[lang_pr])
    return message, os.path.join(tempfile.gettempdir(), f"{name}.npz")


def make_prompt(name, wav, sr, save=True):
    '''
        从音频中提取文本，保存到文件中（可选）
    '''
    global whisper_model
    whisper_model.to(device)
    if not isinstance(wav, torch.FloatTensor):
        wav = torch.tensor(wav)
    if wav.abs().max() > 1:
        wav /= wav.abs().max()
    if wav.size(-1) == 2:
        wav = wav.mean(-1, keepdim=False)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    assert wav.ndim and wav.size(0) == 1
    torchaudio.save(f"./prompts/{name}.wav", wav, sr)
    lang, text = transcribe_one(whisper_model, f"./prompts/{name}.wav")
    lang_token = lang2token[lang]
    text = lang_token + text + lang_token
    with open(f"./prompts/{name}.txt", 'w', encoding='utf-8') as f:
        f.write(text)
    if not save:
        os.remove(f"./prompts/{name}.wav")
        os.remove(f"./prompts/{name}.txt")

    whisper_model.cpu()
    torch.cuda.empty_cache()
    return text, lang
    # text like: [ZH]你好，我是派蒙！[ZH],  lang like: zh   

@torch.no_grad()
def infer_from_audio(text, language, accent, audio_prompt, record_audio_prompt, transcript_content):
    global model, text_collater, text_tokenizer, audio_tokenizer
    audio_prompt = audio_prompt if audio_prompt is not None else record_audio_prompt
    wav_pr, sr = audio_prompt
    if not isinstance(wav_pr, torch.FloatTensor):
        wav_pr = torch.FloatTensor(wav_pr)
    if wav_pr.abs().max() > 1:
        wav_pr /= wav_pr.abs().max()
    if wav_pr.size(-1) == 2:
        wav_pr = wav_pr[:, 0]
    if wav_pr.ndim == 1:
        wav_pr = wav_pr.unsqueeze(0)
    assert wav_pr.ndim and wav_pr.size(0) == 1

    if transcript_content == "":
        text_pr, lang_pr = make_prompt('dummy', wav_pr, sr, save=False)
    else:
        lang_pr = langid.classify(str(transcript_content))[0]
        lang_token = lang2token[lang_pr]
        text_pr = f"{lang_token}{str(transcript_content)}{lang_token}"

    if language == 'auto-detect':
        lang_token = lang2token[langid.classify(text)[0]]
    else:
        lang_token = langdropdown2token[language]
    lang = token2lang[lang_token]
    text = lang_token + text + lang_token

    # onload model
    model.to(device)

    # tokenize audio
    encoded_frames = tokenize_audio(audio_tokenizer, (wav_pr, sr))
    audio_prompts = encoded_frames[0][0].transpose(2, 1).to(device) # (1, T, K)

    # tokenize text
    logging.info(f"synthesize text: {text}")
    phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
    text_tokens, text_tokens_lens = text_collater(
        [
            phone_tokens
        ]
    )
    print("text_tokens: ", text_tokens)

    enroll_x_lens = None
    if text_pr:
        text_prompts, _ = text_tokenizer.tokenize(text=f"{text_pr}".strip())
        text_prompts, enroll_x_lens = text_collater(
            [
                text_prompts
            ]
        )
    text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
    text_tokens_lens += enroll_x_lens
    lang = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]
    print("text_tokens final: ", text_tokens)
    print("lang: ", lang)
    logging.info("Arrive at inference stage (line 285)")
    encoded_frames = model.inference(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        audio_prompts,
        enroll_x_lens=enroll_x_lens,
        top_k=-100,
        temperature=1,
        prompt_language=lang_pr,
        text_language=langs if accent == "no-accent" else lang,
        best_of=5,
    )
    # Decode with Vocos
    frames = encoded_frames.permute(2,0,1)
    features = vocos.codes_to_features(frames) # 反量化 (1, T, 128)
    samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

    # offload model
    model.to('cpu')
    torch.cuda.empty_cache()

    message = f"text prompt: {text_pr}\nsythesized text: {text}"
    return message, (24000, samples.squeeze(0).cpu().numpy())

text = "Hello, I am Paimon!"
language = "auto-detect"
accent = "no-accent"
audio_prompt = torchaudio.load("./prompts/zh-1.wav")
print(audio_prompt)
record_audio_prompt = None
transcript_content = ""
infer_from_audio(text, language, accent, audio_prompt, record_audio_prompt, transcript_content)