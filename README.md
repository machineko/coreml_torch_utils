[![PyTest](https://github.com/machineko/coreml_torch_utils/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/machineko/coreml_torch_utils/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/machineko/coreml_torch_utils/branch/main/graph/badge.svg?token=V1CZl9Uq9i)](https://codecov.io/gh/machineko/coreml_torch_utils)
[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)
[![macOS](https://svgshare.com/i/ZjP.svg)](https://svgshare.com/i/ZjP.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

# Small Coreml utils for deep learning models
### * utils are tested for pytorch but should work for any deep learning framework

## Install
```
pip install coreml-pytorch-utils
```

## Example Usage

```python
from torch import nn
model = nn.Sequential(nn.Conv2d(3, 6, kernel_size=(1, 1)), nn.Conv2d(6, 3, kernel_size=(1, 1)))
model.eval()
jitted = jit.trace(model, example_inputs=(torch.rand(1, 3, 128, 128),))

base_coreml_model = ct.convert(
    jitted,
    inputs=ct.ImageType(
        "name": "x",
        "shape": (1, 3, 128, 128),
        ),
    convert_to="neuralnetwork",
)
utils = [
    InputEnumeratedShapeImage([(64, 128), (128, 256)]),
    OutputDynamicImage(
    height_lower_bound = 64
    height_upper_bound = 128
    width_lower_bound = 128
    width_upper_bound = 256),
    RenameOutput(new_name="SampleOutput")
]
exporter = CoreExporter(utils)
new_model = exporter(base_coreml_model)

# New model output name => SampleOutput, New model output type => Image [Avaliable input shapes 64x128, 128x265]
# Old model output name example => var23, Old model output type Array [Avaliable input shapes 128x128]
```

## Change inputs to images with enumerated shapes
```python
inp_enum = InputEnumeratedShapeImage([(64, 128), (128, 256)]) # [(H W), (H W)]
exporter = CoreExporter([inp_enum])
new_model = exporter(base_coreml_model)
```

## Change output to images with static shape
```python
exporter = CoreExporter([OutputStaticImage(height=128, width=256)])
new_model = exporter(base_coreml_model)
```
