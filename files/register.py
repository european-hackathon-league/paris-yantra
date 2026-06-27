"""Lightweight rigid registration to a canonical reference (SimpleITK, Mattes MI).

Tests finding ②: L2's damage is *independent rigid rotation* of query vs target.
If we rigidly register every volume into ONE common reference frame, that rotation
is undone and MIND/embeddings become viable on L2 again. Cross-modal-safe (MI metric),
no training. This is the cheap proof-of-concept; EasyReg/SynthMorph is the robust upgrade.
"""
from __future__ import annotations
import torch


def _to_sitk(vol: torch.Tensor):
    import SimpleITK as sitk
    arr = vol.squeeze(0).detach().cpu().float().numpy()      # [D,H,W]
    return sitk.Cast(sitk.GetImageFromArray(arr), sitk.sitkFloat32)


def _to_tensor(img) -> torch.Tensor:
    import SimpleITK as sitk
    return torch.from_numpy(sitk.GetArrayFromImage(img)).unsqueeze(0).float()


def register_to_ref(moving: torch.Tensor, fixed: torch.Tensor, iters: int = 80) -> torch.Tensor:
    """Rigidly register `moving` [1,R,R,R] onto `fixed` [1,R,R,R] frame; return resampled moving."""
    import SimpleITK as sitk
    f, m = _to_sitk(fixed), _to_sitk(moving)
    init = sitk.CenteredTransformInitializer(
        f, m, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)   # cross-modal-safe
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.1)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=iters)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SetInitialTransform(init, inPlace=False)
    try:
        t = R.Execute(f, m)
    except Exception:
        t = init                                              # fall back to centering if it diverges
    out = sitk.Resample(m, f, t, sitk.sitkLinear, 0.0, m.GetPixelID())
    return _to_tensor(out)
