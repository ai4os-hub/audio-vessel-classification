# audio_vessel_classifier
[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-hub/audio-vessel-classification/main)](https://jenkins.services.ai4os.eu/job/AI4OS-hub/job/audio-vessel-classification/job/main/)

The weights of the models (fine-tuning & feature-extraction) can be requested by emailing: wout.decrop@vliz.be
The weights are currently not open-source untill the paper (in submission) is accepted

To launch it, first install the package then run [deepaas](https://github.com/ai4os/DEEPaaS):
```bash
git clone https://github.com/ai4os-hub/audio_vessel_classifier
cd audio_vessel_classifier
pip install -e .
deepaas-run --listen-ip 0.0.0.0
```

You will have the option to upload either a (10 sec) .wav file or  embedding file (.pt)
You can then choose if you want to use the feature-extraction or fine-tuning model. The fine-tuning model has a slightly higher performance and is most recommended.

## Project structure
```
│
├── Dockerfile             <- Describes main steps on integration of DEEPaaS API and
│                             audio_vessel_classifier application in one Docker image
│
├── Jenkinsfile            <- Describes basic Jenkins CI/CD pipeline (see .sqa/)
│
├── LICENSE                <- License file
│
├── README.md              <- The top-level README for developers using this project.
│
├── VERSION                <- audio_vessel_classifier version file
│
├── .sqa/                  <- CI/CD configuration files
│
├── audio_vessel_classifier    <- Source code for use in this project.
│   │
│   ├── __init__.py        <- Makes audio_vessel_classifier a Python module
│   │
│   ├── api.py             <- Main script for the integration with DEEPaaS API
│   |
│   ├── config.py          <- Configuration file to define Constants used across audio_vessel_classifier
│   │
│   └── misc.py            <- Misc functions that were helpful accross projects
│
├── data/                  <- Folder to store the data
│
├── models/                <- Folder to store models
│   
├── tests/                 <- Scripts to perfrom code testing
|
├── metadata.json          <- Metadata information propagated to the AI4OS Hub
│
├── pyproject.toml         <- a configuration file used by packaging tools, so audio_vessel_classifier
│                             can be imported or installed with  `pip install -e .`                             
│
├── requirements.txt       <- The requirements file for reproducing the analysis environment, i.e.
│                             contains a list of packages needed to make audio_vessel_classifier work
│
├── requirements-test.txt  <- The requirements file for running code tests (see tests/ directory)
│
└── tox.ini                <- Configuration file for the tox tool used for testing (see .sqa/)
```
