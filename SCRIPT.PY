# -*- coding: utf-8 -*-
"""
Functions to integrate your model with the DEEPaaS API.
It's usually good practice to keep this file minimal, only performing
the interfacing tasks. In this way you don't mix your true code with
DEEPaaS code and everything is more modular. That is, if you need to write
the predict() function in api.py, you would import your true predict function
and call it from here (with some processing / postprocessing in between
if needed).
For example:

    import mycustomfile

    def predict(**kwargs):
        args = preprocess(kwargs)
        resp = mycustomfile.predict(args)
        resp = postprocess(resp)
        return resp

To start populating this file, take a look at the docs [1] and at
an exemplar module [2].

[1]: https://docs.ai4os.eu/
[2]: https://github.com/ai4os-hub/ai4os-demo-app
"""


import builtins
import json
import logging

import torch
import torch.nn.functional as F
import torchaudio
from aiohttp.web import HTTPException
from transformers import ClapAudioModelWithProjection, ClapProcessor
from webargs import fields

# from audio_vessel_classifier import config
from audio_vessel_classifier.models import model_loader

# logger = logging.getLogger(__name__)
# logger.setLevel(config.LOG_LEVEL)



def load_and_prepare_waveform(
    wav_path, duration=10, desired_fs=48000, channel=0
):
    waveform_info = torchaudio.info(wav_path)
    waveform, fs = torchaudio.load(wav_path)

    if waveform_info.sample_rate != desired_fs:
        resampler = torchaudio.transforms.Resample(fs, desired_fs)
        waveform = resampler(waveform)

    max_samples = int(duration * desired_fs)
    waveform = waveform[channel, :max_samples]

    if waveform.shape[0] < max_samples:
        waveform = F.pad(waveform, (0, max_samples - waveform.shape[0]))

    return waveform.cpu().numpy(), desired_fs


def get_clap_embedding(x_np, desired_fs, freeze, device):
    processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual")
    clap = ClapAudioModelWithProjection.from_pretrained(
        "davidrrobinson/BioLingual"
    ).to(device)

    inputs = processor(
        audios=[x_np],
        return_tensors="pt",
        sampling_rate=desired_fs,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if freeze:
            return clap(**inputs).audio_embeds
        else:
            return inputs["input_features"]


def run_prediction(embedding, model, freeze, device):
    model.eval()
    with torch.no_grad():
        embedding = embedding.to(device)
        output = model(embedding)[0] if freeze else model(embedding)
        predicted_class = torch.argmax(output, dim=1).item()
        probabilities = F.softmax(output, dim=1).squeeze().cpu().tolist()
    return predicted_class, probabilities


def predict_data(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freeze = args.get("model_choice") != "fine_tuning"
    embedded = False

    if args.get("embedding_file"):
        embedded = True

    if args.get("audio_file"):
        # Extract the temp file path
        wav_path = wav_path = args["audio_file"].filename
        # path from args, not hardcoded
        x_np, desired_fs = load_and_prepare_waveform(wav_path)
        embedding = get_clap_embedding(x_np, desired_fs, freeze, device)
        embedded = True

    if embedded:
        logger.debug("Predict with args: %s", args)
        try:
            update_with_query_conf(args)
            config.conf_dict

            if "embedding_file" in args and args["embedding_file"] is not None:
                uploaded_file = args["embedding_file"]
                embedding = torch.load(
                    uploaded_file.filename, map_location="cpu"
                )

            model = model_loader(device, freeze)
            return run_prediction(embedding, model, freeze, device)

        except Exception as e:
            logger.exception("Error during prediction")
            return {"error": str(e)}
    else:
        return {"error": "No audio or embedding file provided"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
freeze=True
model = model_loader(device, freeze)


import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === CONFIGURATION ===
wav_dir = "/srv/audio-vessel-classification/data/processed_folder"

# Bins: 0–1 km, 1–2 km, ..., 9–10 km, >10 km
bin_edges = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, float("inf")]
bin_labels = list(range(len(bin_edges) - 1))

# === GATHER ALL WAV FILES ===
all_wav_paths = []
for root, dirs, files in os.walk(wav_dir):
    for filename in files:
        if filename.endswith(".wav"):
            all_wav_paths.append(os.path.join(root, filename))

# === PROCESS FILES ===
results = []

for wav_path in tqdm(all_wav_paths, desc="Processing WAV files"):
    try:
        # STEP 1: Load waveform
        x_np, desired_fs = load_and_prepare_waveform(wav_path)

        # STEP 2: Get CLAP embedding
        embedding = get_clap_embedding(x_np, desired_fs, freeze, device)

        # STEP 3: Run prediction
        prediction = run_prediction(embedding, model, freeze, device)  # expected to return (class_index, prob_list)
        predicted_class = prediction[0]
        probs = prediction[1]

        # STEP 4: Parse actual distance
        base = os.path.splitext(os.path.basename(wav_path))[0]
        dist_str = base.split("_")[-1]  # last element
        actual_distance = int(dist_str)  # in meters

        # STEP 5: Bin actual distance
        true_class = pd.cut([actual_distance], bins=bin_edges, labels=bin_labels)[0]

        # STEP 6: Save + print
        print(f"{wav_path}: predicted class = {predicted_class}, actual distance = {actual_distance} m")

        results.append({
            "file": wav_path,
            "true_class": int(true_class),
            "predicted_class": int(predicted_class),
            "actual_distance_m": actual_distance,
            "probabilities": probs
        })

    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        continue

# === SAVE TO CSV ===
df = pd.DataFrame(results)
df.to_csv("distance_predictions.csv", index=False)
print("✅ Results saved to distance_predictions.csv")

# === CONFUSION MATRIX ===
cm = confusion_matrix(df["true_class"], df["predicted_class"], labels=bin_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{i}-{i+1}" for i in range(10)] + [">10"])
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix (Predicted vs Actual Class)")
plt.tight_layout()
plt.show()
