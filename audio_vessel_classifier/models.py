import torch
import torch.nn as nn
from transformers import ClapAudioModelWithProjection


class Fine_tuning_CLAPModel(nn.Module):
    def __init__(self, clap_model, linear_model):
        super().__init__()
        self.clap_model = clap_model
        self.linear_model = linear_model

    def forward(self, x):
        if x.dim() == 2:  # (features, time)
            x = x.unsqueeze(0)

        with torch.no_grad():
            output = self.clap_model(x.to(x.device))

        embeddings = output.audio_embeds
        logits = self.linear_model(embeddings)
        return logits


class Feature_extraction_CLAPModel(nn.Module):
    def __init__(self, fc1, fc2, relu):
        super().__init__()
        self.fc1 = fc1
        self.fc2 = fc2
        self.relu = relu

    def forward(self, x):
        if x.dim() == 2:  # (features, time)
            x = x.unsqueeze(0)

        with torch.no_grad():
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
        return out


def model_loader(device, freeze=True):
    if freeze:
        fc1 = nn.Linear(in_features=512, out_features=256)
        fc2 = nn.Linear(in_features=256, out_features=11)
        relu = nn.ReLU()

        fc1_path = (
            "/srv/DEEP-OC-underwater-noise-classification/"
            "models/feature_extraction/model/fc1.pth"
        )
        fc1.load_state_dict(
            torch.load(
                fc1_path,
                map_location=device,
            )
        )

        fc2_path = (
            "/srv/DEEP-OC-underwater-noise-classification/"
            "models/feature_extraction/model/fc2.pth"
        )
        fc2.load_state_dict(
            torch.load(
                fc2_path,
                map_location=device,
            )
        )

        model = Feature_extraction_CLAPModel(fc1, fc2, relu).to(device)
        model.eval()
        return model

    else:
        clap_model_path = (
            "/srv/DEEP-OC-underwater-noise-classification/"
            "models/fine_tuning/model"
        )
        clap_model = ClapAudioModelWithProjection.from_pretrained(
            clap_model_path,
            revision="808bb50859ce7d0c0fcc2b233676c7ba9319107e"
        ).to(device)

        clap_model.eval()

        # Load linear head
        linear_model = nn.Linear(in_features=512, out_features=11)
        linear_pth = (
            "/srv/DEEP-OC-underwater-noise-classification/"
            "models/fine_tuning/model/linear.pth"
        )
        linear_model.load_state_dict(
            torch.load(
                linear_pth,
                map_location=device,
            )
        )
        linear_model = linear_model.to(device)

        model = Fine_tuning_CLAPModel(clap_model, linear_model).to(device)
        model.eval()
        return model
