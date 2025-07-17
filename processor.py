"""
Inference script for predicting malignancy of lung nodules
"""
import numpy as np
import dataloader
import torch
import torch.nn as nn
from torchvision import models
from models.model_3d import I3D
from models.model_2d import ResNet18
from model import ConvNextLSTM
import os
import math
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

# define processor
class MalignancyProcessor:
    """
    Loads a chest CT scan, and predicts the malignancy around a nodule
    """

    def __init__(self, mode="ConvNextLSTM", suppress_logs=False, model_name="LUNA25-baseline-ConvNextLSTM"):

        self.size_px = 64
        self.size_mm = 50

        self.model_name = model_name
        self.mode = mode
        self.suppress_logs = suppress_logs

        if not self.suppress_logs:
            logging.info("Initializing the deep learning system")

        if self.mode == "2D":
            self.model_2d = ResNet18(weights=None).cuda()
        elif self.mode == "3D":
            self.model_3d = I3D(num_classes=1, pre_trained=False, input_channels=3).cuda()
        elif self.mode == "ConvNextLSTM":
            self.model_convnext = ConvNextLSTM(pretrained=False, in_chans=3, class_num=1).cuda()

        # Auto-detect environment and set model root path
        import os
        if os.path.exists("/opt/app/resources"):
            # Docker container environment
            self.model_root = "/opt/app/resources"
        else:
            # Local testing environment
            self.model_root = "./resources"

    def define_inputs(self, image, header, coords):
        self.image = image
        self.header = header
        self.coords = coords

    def extract_patch(self, coord, output_shape, mode):

        patch = dataloader.extract_patch(
            CTData=self.image,
            coord=coord,
            srcVoxelOrigin=self.header["origin"],
            srcWorldMatrix=self.header["transform"],
            srcVoxelSpacing=self.header["spacing"],
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            coord_space_world=True,
            mode=mode,
        )

        # ensure same datatype...
        patch = patch.astype(np.float32)

        # clip and scale...
        patch = dataloader.clip_and_scale(patch)
        return patch

    def _process_model(self, mode):

        if not self.suppress_logs:
            logging.info("Processing in " + mode)

        if mode == "2D":
            output_shape = [1, self.size_px, self.size_px]
            model = self.model_2d
        elif mode == "3D":
            output_shape = [self.size_px, self.size_px, self.size_px]
            model = self.model_3d
        elif mode == "ConvNextLSTM":
            # ConvNextLSTM expects 3D volume data
            output_shape = [self.size_px, self.size_px, self.size_px]
            model = self.model_convnext

        nodules = []

        for _coord in self.coords:
            patch = self.extract_patch(_coord, output_shape, mode=mode)
            nodules.append(patch)

        nodules = np.array(nodules)
        
        if mode == "ConvNextLSTM":
            # ConvNextLSTM expects input shape (bs, in_chans, n_slice_per_c, image_size, image_size)
            # Convert 3D volume to multi-channel format
            bs = nodules.shape[0]
            
            # Process each nodule
            nodules_slices = []
            for i in range(bs):
                volume = nodules[i]  # (1, 64, 64, 64) for 3D mode
                if volume.ndim == 4 and volume.shape[0] == 1:
                    volume = volume.squeeze(0)  # Remove first dimension -> (64, 64, 64)
                
                # Convert to (3, depth, height, width) by repeating channels
                volume_rgb = np.stack([volume, volume, volume], axis=0)  # (3, 64, 64, 64)
                nodules_slices.append(volume_rgb)
            
            nodules = np.array(nodules_slices)  # (bs, 3, 64, 64, 64)
            
        nodules = torch.from_numpy(nodules).cuda()

        # Load model weights
        if mode == "ConvNextLSTM":
            ckpt = torch.load(os.path.join(self.model_root, "best_metric_model.pth"))
        else:
            ckpt = torch.load(
                os.path.join(
                    self.model_root,
                    self.model_name,
                    "best_metric_model.pth",
                )
            )
        model.load_state_dict(ckpt)
        model.eval()
        logits = model(nodules)
        logits = logits.data.cpu().numpy()

        logits = np.array(logits)
        return logits

    def predict(self):

        logits = self._process_model(self.mode)

        probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return probability, logits