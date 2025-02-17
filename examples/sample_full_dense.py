"""
Example of using FullImageDenseSampler.
"""

from pathlib import Path

from patch_samplers.full_samplers import (
    FullImageDenseSampler,
    SamplerExecutionMode,
)


if __name__ == "__main__":
    img_path = Path(
        "/home/xubiker/dev/PATH-DT-MSU.WSS2/images/test/test_01.psi"
    )

    patch_sampler = FullImageDenseSampler(
        img_path,
        layer=2,
        patch_size=224,
        batch_size=16,
        stride=112,
        mode=SamplerExecutionMode.INMEMORY_SINGLEPROC,
    )

    for inputs, coords, filled_ratio in patch_sampler.generator_torch():
        print(inputs.shape, coords.shape, filled_ratio)
