"""
This file gathers some functions that have proven to be useful
across projects. They are not strictly need for integration
but users might want nevertheless to take advantage from them.
"""

import subprocess  # nosec
import warnings
from functools import wraps
from multiprocessing import Process

from aiohttp.web import HTTPBadRequest


def _catch_error(f):
    """
    Decorate API functions to return an error as HTTPBadRequest,
    in case it fails.
    """

    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)

    return wrap


def _fields_to_dict(fields_in):
    """
    Function to convert marshmallow fields to dict()
    """
    dict_out = {}
    for k, v in fields_in.items():
        param = {}
        param["default"] = v.missing
        param["type"] = type(v.missing)
        param["required"] = getattr(v, "required", False)

        v_help = v.metadata["description"]
        if "enum" in v.metadata.keys():
            v_help = f"{v_help}. Choices: {v.metadata['enum']}"
        param["help"] = v_help

        dict_out[k] = param

    return dict_out


def mount_nextcloud(frompath, topath):
    """
    Mount a NextCloud folder in your local machine or viceversa.

    Example of usage:
        mount_nextcloud('rshare:/data/images', 'my_local_image_path')

    Parameters
    ==========
    * frompath: str, pathlib.Path
        Source folder to be copied
    * topath: str, pathlib.Path
        Destination folder
    """
    command = ["rclone", "copy", f"{frompath}", f"{topath}"]
    result = subprocess.Popen(
        command,  # nosec
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output, error = result.communicate()
    if error:
        warnings.warn(f"Error while mounting NextCloud: {error}")
    return output, error


def launch_cmd(logdir, port):
    subprocess.call(
        [
            "tensorboard",  # nosec
            "--logdir",
            f"{logdir}",
            "--port",
            f"{port}",
            "--host",
            "0.0.0.0",
        ]
    )


def launch_tensorboard(logdir, port=6006):
    """
    Run Tensorboard on a separate Process on behalf of the user

    Parameters
    ==========
    * logdir: str, pathlib.Path
        Folder path to tensorboard logs.
    * port: int
        Port to use for the monitoring webserver.
    """
    subprocess.run(  # nosec
        # kill any other process in that port
        ["fuser", "-k", f"{port}/tcp"]  # nosec
    )
    p = Process(target=launch_cmd, args=(logdir, port), daemon=True)
    p.start()
