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

from audio_vessel_classifier import config
from audio_vessel_classifier.models import model_loader

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def get_metadata():
    """Returns a dictionary containing metadata information about the module.
       DO NOT REMOVE - All modules should have a get_metadata() function

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("Collecting metadata from: %s", config.API_NAME)
        metadata = config.PROJECT_METADATA
        # TODO: Add dynamic metadata collection here
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        logger.error("Error collecting metadata: %s", err, exc_info=True)
        raise  # Reraise the exception after log


def predict(**args):
    logger.debug("Predict with args: %s", args)
    try:
        if not any([args.get("embedding_file"), args.get("audio_file")]):

            raise Exception("You must provide  '.pt' or '.wav' in the payload!")

        return predict_data(args)

    except Exception as err:
        raise HTTPException(reason=err) from err


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
    # Get this hash from the model card under 'Files and versions' (use commit SHA)
    revision = "808bb50859ce7d0c0fcc2b233676c7ba9319107e"  

    processor = ClapProcessor.from_pretrained(
        "davidrrobinson/BioLingual",
        revision="808bb50859ce7d0c0fcc2b233676c7ba9319107e"
    )
    clap = ClapAudioModelWithProjection.from_pretrained(
        "davidrrobinson/BioLingual",
        revision="808bb50859ce7d0c0fcc2b233676c7ba9319107e"
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


def update_with_query_conf(user_args):
    """
    Update the default YAML configuration
    with the user's input args from the API query
    """
    # Update the default conf with the user input
    CONF = config.CONF
    for group, val in sorted(CONF.items()):
        for g_key, g_val in sorted(val.items()):
            if g_key in user_args:
                raw_value = user_args[g_key]
                if not raw_value:
                    continue  # skip if the value is empty
                try:
                    # Try parsing as JSON
                    g_val["value"] = json.loads(raw_value)
                except json.JSONDecodeError:
                    # Fall back to treating it as a plain string
                    g_val["value"] = raw_value
    # Check and save the configuration
    config.check_conf(conf=CONF)
    config.conf_dict = config.get_conf_dict(conf=CONF)


def get_predict_args():
    """
    TODO: add more dtypes
    * int with choices
    * composed: list of strs, list of int
    """
    # WARNING: missing!=None has to go with required=False
    # fmt: off
    arg_dict = {
        "model_choice": fields.Str(
            required=False,
            missing="fine_tuning",
            enum=["fine_tuning", "feature_extraction"],
            description="test multi-choice with strings",
        ),

        "embedding_file": fields.Field(
            required=False,
            format="binary",
            type="file",
            location="form",
            description="test image upload",
        ),
        "audio_file": fields.Field(
            required=False,
            type="file",
            location="form",
            description="test audio upload",
        ),
    }
    # fmt: on
    return arg_dict


def populate_parser(parser, default_conf):
    """
    Returns a arg-parse like parser.
    """
    print("popu parser")
    for group, val in default_conf.items():
        for g_key, g_val in val.items():
            gg_keys = g_val.keys()

            # Load optional keys
            help = g_val["help"] if ("help" in gg_keys) else ""
            type = (
                getattr(builtins, g_val["type"])
                if ("type" in gg_keys)
                else None
            )
            choices = g_val["choices"] if ("choices" in gg_keys) else None

            # Additional info in help string
            help += "\n" + "<font color='#C5576B'> Group name: **{}**".format(
                str(group)
            )
            if choices:
                help += "\n" + "Choices: {}".format(str(choices))
            if type:
                help += "\n" + "Type: {}".format(g_val["type"])
            help += "</font>"

            # Create arg dict
            opt_args = {
                "load_default": json.dumps(g_val["value"]),
                "metadata": {"description": help},
                "required": False,
            }
            if choices:
                json_choices = [json.dumps(i) for i in choices]
                opt_args["metadata"]["enum"] = json_choices
                opt_args["validate"] = fields.validate.OneOf(json_choices)
            parser[g_key] = fields.Str(**opt_args)

    return parser
