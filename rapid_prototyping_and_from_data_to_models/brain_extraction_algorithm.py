import imfusion
import numpy as np
import torch
from HD_BET.hd_bet_prediction import get_hdbet_predictor

@imfusion.data.components.register(name="quality_control")
class QualityControl:
    pose_change: np.ndarray | None = imfusion.data.components.field()
    lc2: float = imfusion.data.components.field()
    inside_brain: float = imfusion.data.components.field()

    def __init__(self, pose_change: np.ndarray | None = None, lc2: float = 0.0, inside_brain: float = 0.0):
        self.pose_change = pose_change
        self.lc2 = lc2
        self.inside_brain = inside_brain


class BrainExtraction:
    def __init__(self, spacing: float = 2.0):
        self.spacing = spacing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = get_hdbet_predictor(device=device)

    def __call__(self, image: imfusion.SharedImageSet) -> imfusion.SharedImageSet:
        # The check does not need a mask at full resolution.
        image = imfusion.processing.resample_to_spacing(image.clone(), [self.spacing] * 3)

        volume = image.astype(np.float32).numpy().squeeze()
        mask = self.predictor.predict_single_npy_array(
            volume[None], {"spacing": [self.spacing] * 3}, None, None, False
        )

        # Casting the resampled MR keeps its spacing and pose for the mask.
        brain = image.astype(np.uint8)
        brain[0].assign_array(mask[..., None])
        brain.modality = imfusion.Data.Modality.LABEL
        return brain


@imfusion.algorithm.register(display_name="MICCAI;Brain Extraction (HD-BET)")
class BrainExtractionAlgorithm:
    image = imfusion.algorithm.Input(imfusion.SharedImageSet, validator=lambda sis: len(sis) == 1)
    spacing = imfusion.algorithm.ParamDouble("Spacing", default=2.0, min=0.5, max=5.0, unit="mm", with_slider=True)

    def __call__(self) -> imfusion.SharedImageSet:
        return BrainExtraction(self.spacing)(self.image)