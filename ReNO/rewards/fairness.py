import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from facenet_pytorch import MTCNN

from rewards.base_reward import BaseRewardLoss

# Same normalization constants as rewards/utils.py's clip_img_transform and
# the FairFace preprocessing in scripts/racial_composition_analyze.py.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

RACE_LABELS = [
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
]


class FairnessLoss(BaseRewardLoss):
    """
    Penalizes deviation of the FairFace race classifier's predicted
    probability distribution -- on the image being generated right now --
    from a target distribution (uniform over the 7 FairFace race categories
    by default). This gets a real gradient signal about race every
    iteration, from the same classifier used to audit racial composition
    (scripts/racial_composition_analyze.py), unlike BiasCorrector
    (training/bias_correction.py) which nudges latents along a static
    direction that was never validated to correlate with race.

    Face crop: MTCNN itself isn't differentiable, so it's used only to
    localize a bounding box (under no_grad, on a detached copy of the
    current image). That box is then used to crop+resize the *graph-
    connected* tensor with F.interpolate, which is differentiable, so
    gradients still flow from the classifier back to the latents. If no
    face is detected, falls back to the full image.

    Adversarial-robustness defense: gradient ascent against a frozen
    classifier will happily find imperceptible high-frequency pixel
    patterns that flip its prediction without changing the depicted race
    (measured directly: KL loss converged to ~0 while an independent
    MTCNN+classifier audit of the saved PNGs showed no change in the
    actual face distribution). A cheap blur + downsample/upsample before
    classification destroys that high-frequency channel, so gradients can
    only exploit genuine, low-frequency (i.e. visible) image structure.
    """

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        fairface_weights: str,
        memsave: bool = False,
        target_dist=None,
    ):
        model = torchvision.models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 18)
        state_dict = torch.load(fairface_weights, map_location="cpu")
        model.load_state_dict(state_dict)
        self.model = model.to(device, dtype=dtype).eval()
        self.freeze_parameters(self.model.parameters())
        self.memsave = memsave
        # CPU: MTCNN's tiny-kernel convs hit a cuDNN "unable to find an
        # engine" error on some GPU/driver combos at small pyramid scales
        # (same issue worked around in racial_composition_analyze_by_stage.py).
        self.mtcnn = MTCNN(
            image_size=224, margin=int(224 * 0.25), post_process=False, device="cpu"
        )
        if memsave:
            import memsave_torch.nn

            self.model = memsave_torch.nn.convert_to_memory_saving(self.model).to(
                device, dtype=dtype
            )

        if target_dist is None:
            target_dist = [1.0 / len(RACE_LABELS)] * len(RACE_LABELS)
        assert len(target_dist) == len(RACE_LABELS)
        self.target_logprobs = torch.tensor(
            target_dist, device=device, dtype=torch.float32
        ).log()

        self.clip_mean = torch.tensor(CLIP_MEAN, device=device, dtype=dtype).view(
            1, 3, 1, 1
        )
        self.clip_std = torch.tensor(CLIP_STD, device=device, dtype=dtype).view(
            1, 3, 1, 1
        )
        self.imagenet_mean = torch.tensor(
            IMAGENET_MEAN, device=device, dtype=dtype
        ).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(
            1, 3, 1, 1
        )

        super().__init__("Fairness", weighting)

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("FairnessLoss overrides __call__ directly.")

    def get_text_features(self, prompt: str) -> torch.Tensor:
        raise NotImplementedError("FairnessLoss overrides __call__ directly.")

    def compute_loss(self, image_features, text_features) -> torch.Tensor:
        raise NotImplementedError("FairnessLoss overrides __call__ directly.")

    def _crop_to_face(self, pixels: torch.Tensor) -> torch.Tensor:
        b, _, h, w = pixels.shape
        crops = []
        for i in range(b):
            with torch.no_grad():
                img_np = (
                    pixels[i].detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                    * 255
                ).astype(np.uint8)
                boxes, _ = self.mtcnn.detect(img_np)
            crop = pixels[i : i + 1]
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = boxes[0]
                x1, y1 = max(int(x1), 0), max(int(y1), 0)
                x2, y2 = min(int(x2), w), min(int(y2), h)
                if x2 - x1 >= 8 and y2 - y1 >= 8:
                    crop = F.interpolate(
                        crop[:, :, y1:y2, x1:x2],
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )
            crops.append(crop)
        return torch.cat(crops, dim=0)

    def _blur_defense(self, pixels: torch.Tensor) -> torch.Tensor:
        # A *fixed* blur is fully differentiable, so gradient ascent just
        # finds a noise pattern tuned to survive that exact transform
        # (adversarial "gradient masking" -- confirmed empirically: at
        # fairness_weighting=15 the KL loss collapsed and the classifier's
        # predicted race flipped, but an independent visual check of the
        # saved PNGs showed no actual change in the depicted face).
        # Randomizing the downsample factor and adding pixel noise each
        # call means no single fixed perturbation survives every step --
        # the optimizer can only get consistent reward from changes that
        # remain visible after an unpredictable blur, i.e. genuine,
        # low-frequency (real) image structure. Same idea as Expectation-
        # over-Transformation, applied against the attacker instead of by it.
        b, c, h, w = pixels.shape
        pixels = pixels + 0.02 * torch.randn_like(pixels)
        pixels = F.avg_pool2d(pixels, kernel_size=3, stride=1, padding=1)
        factor = int(torch.randint(2, 5, (1,)).item())  # random in {2,3,4}
        small = F.interpolate(
            pixels, size=(h // factor, w // factor), mode="bilinear", align_corners=False
        )
        return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)

    def __call__(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        if self.memsave:
            image = image.to(torch.float32)
        # `image` arrives CLIP-normalized + resized to 224 (trainer.py's
        # shared preprocess_fn). Undo that to recover ~[0,1] pixels, then
        # apply the ImageNet normalization FairFace's resnet34 expects.
        pixels = image * self.clip_std.to(image.dtype) + self.clip_mean.to(image.dtype)

        pixels = self._crop_to_face(pixels)
        pixels = self._blur_defense(pixels)

        fairface_input = (pixels - self.imagenet_mean.to(image.dtype)) / (
            self.imagenet_std.to(image.dtype)
        )

        with torch.autocast("cuda"):
            logits = self.model(fairface_input)[:, :7]
        race_logprobs = torch.log_softmax(logits.float(), dim=-1)
        race_probs = race_logprobs.exp()

        # KL(predicted || target): pushes the predicted race distribution
        # toward target_dist (uniform by default).
        kl = (race_probs * (race_logprobs - self.target_logprobs)).sum(dim=-1).mean()
        return kl
