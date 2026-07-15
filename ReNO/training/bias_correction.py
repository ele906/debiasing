import torch


class BiasCorrector:
    """
    Counteracts the systematic latent-space drift that reward-based noise
    optimization (ReNO) tends to induce, by pulling latents back along a
    pre-computed "bias direction" (see scripts/estimate_bias_direction.py).

    The correction is scaled rather than a one-off fixed offset: at every
    iteration it removes a fraction (`scale`) of however far the latents have
    already drifted from their initial value along that direction, so the
    push grows and shrinks with the actual amount of drift instead of always
    being the same size.
    """

    def __init__(
        self,
        direction_path: str,
        scale: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        direction = torch.load(direction_path, map_location="cpu").to(
            device=device, dtype=dtype
        )
        self.direction = direction / direction.norm()
        self.scale = scale

    def step(self, latents: torch.Tensor, init_latents: torch.Tensor) -> float:
        """Nudge `latents` (in place) back along the bias direction.

        Returns the signed drift projection (before correction) for logging.
        """
        with torch.no_grad():
            drift = (latents.detach() - init_latents).flatten()
            direction_flat = self.direction.flatten()
            projection = torch.dot(drift, direction_flat)
            correction = (-self.scale * projection) * self.direction
            latents.data.add_(correction)
        return projection.item()
