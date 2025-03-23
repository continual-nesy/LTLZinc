import torch
from torchvision.models import GoogLeNetOutputs
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5
from torchvision.models.densenet import densenet121
from torchvision.models.squeezenet import squeezenet1_1
from torchvision.models.googlenet import googlenet

class Backbone(torch.nn.Module):
    """
    Generic backbone module. It initializes the perceptual backbone, possibly downloading pre-trained weights.
    """
    def __init__(self, opts):
        super().__init__()

        if opts["pretrained_weights"]:
            # Load ImageNet weights and replace each model's classifier with an emb_size linear layer.
            if opts["backbone_module"] == "squeezenet11":
                self.model = squeezenet1_1(weights="DEFAULT")
                final_conv = torch.nn.Conv2d(512, opts["emb_size"], kernel_size=1)
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.5), final_conv, torch.nn.ReLU(inplace=True), torch.nn.AdaptiveAvgPool2d((1, 1)))
                torch.nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
            elif opts["backbone_module"] == "shufflenetv2x05":
                self.model = shufflenet_v2_x0_5(weights="DEFAULT")
                self.model.fc = torch.nn.Linear(self.model._stage_out_channels[-1], opts["emb_size"])
            elif opts["backbone_module"] == "googlenet":
                self.model = googlenet(weights="DEFAULT")
                self.model.fc = torch.nn.Linear(1024, opts["emb_size"])
            elif opts["backbone_module"] == "densenet121":
                self.model = densenet121(weights="DEFAULT")
                self.model.classifier = torch.nn.Linear(1024, opts["emb_size"])
                torch.nn.init.constant_(self.model.classifier.bias, 0)
        else:
            if opts["backbone_module"] == "squeezenet11":
                self.model = squeezenet1_1(num_classes=opts["emb_size"])
            elif opts["backbone_module"] == "shufflenetv2x05":
                self.model = shufflenet_v2_x0_5(num_classes=opts["emb_size"])
            elif opts["backbone_module"] == "googlenet":
                self.model = googlenet(num_classes=opts["emb_size"])
            elif opts["backbone_module"] == "densenet121":
                self.model = densenet121(num_classes=opts["emb_size"])

    def forward(self, x):
        return self.model(x)

class ContinualClassifier(torch.nn.Module):
    """
     Continual learning classifier, it contains a perceptual component (self.backbone), a dictionary of hidden layers
     (self.hidden_layers), which are possibly instantiated as a modular architecture, and a dictionary of classification
     heads (self.heads).
     The mapping between the hidden layers and the classification heads reflects the type of available knowledge.
    """
    def __init__(self, opts, classes, knowledge_bins):
        super().__init__()

        self.backbone = Backbone(opts)
        self.modular = opts["modular_net"] and "none" not in knowledge_bins # If no knowledge is available, override modular network with a flat one.

        self.knowledge_bins = knowledge_bins
        self.one_to_one_mapping = "none" in knowledge_bins

        num_inputs = len(classes.keys())

        self.hidden_layers = torch.nn.ModuleDict({k: torch.nn.Sequential(
            torch.nn.Linear(opts["emb_size"] * num_inputs, opts["hidden_size"]), torch.nn.ReLU()) for k in
                                                  self.knowledge_bins})

        if self.one_to_one_mapping:
            self.hidden_size = opts["hidden_size"]
        else:
            self.hidden_size = opts["hidden_size"] * len(self.hidden_layers)

        self.heads = torch.nn.ModuleDict({k: torch.nn.Linear(self.hidden_size, len(v)) for k, v in classes.items()})

    def forward(self, imgs):
        batched_imgs = torch.flatten(imgs, start_dim=0, end_dim=1)
        batched_embs = self.backbone(batched_imgs)
        if isinstance(batched_embs, GoogLeNetOutputs):
            batched_embs = batched_embs.logits
            
        bb_embs = torch.reshape(batched_embs, (imgs.size(0), -1))

        if self.one_to_one_mapping:
            hidden_embs = {k: v(bb_embs) for k, v in self.hidden_layers.items()}
            if "none" in hidden_embs.keys():
                logits = {k: v(hidden_embs["none"]) for k, v in self.heads.items()}
                logits = torch.stack([logits[k] for k in sorted(self.heads.keys())], dim=-1).squeeze()
            else:
                logits = {k: v(hidden_embs[k]) for k, v in self.heads.items()}
                logits = torch.stack([logits[k] for k in sorted(self.heads.keys())], dim=-1).squeeze()

        else:  # 1:n or m:n mapping
            hidden_embs = torch.cat([self.hidden_layers[k](bb_embs) for k in sorted(self.hidden_layers.keys())], dim=-1)
            logits = torch.stack([self.heads[k](hidden_embs) for k in sorted(self.heads.keys())],
                                 dim=-1).squeeze()  # (batch, labels, output)

        return logits

    def train_layers(self, layers_list):
        """
        Set all the intermediate layers in the list as trainable and everything else as non trainable.
        :param layer_list: list of  hidden layer names.
        """
        for n, l in self.hidden_layers.items():
            l.requires_grad_(n in layers_list)