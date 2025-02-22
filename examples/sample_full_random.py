"""
Example of using FullImageRandomSampler.
"""

from pathlib import Path

from patch_samplers.full_samplers import (
    FullImageRndSampler,
    SamplerExecutionMode,
)


if __name__ == "__main__":

    img_path = Path(
        "/home/xubiker/dev/PATH-DT-MSU.WSS2/images/test/test_01.psi"
    )

    patch_sampler = FullImageRndSampler(
        img_path,
        layer=2,
        patch_size=224,
        batch_size=16,
        mode=SamplerExecutionMode.INMEMORY_SINGLEPROC,
    )

    for inputs, coords, filled_ratio in patch_sampler.generator_torch():
        print(inputs.shape, coords.shape, filled_ratio)
