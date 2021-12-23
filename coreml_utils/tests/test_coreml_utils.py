import pytest
from coreml_utils.utils import (
    RenameOutput,
    InputEnumeratedShapeImage,
    CoreExporter,
    OutputDynamicImage,
    OutputStaticImage,
)
import torch.nn as nn
import coremltools as ct
from torch import jit
import torch


import torch.nn as nn
import coremltools as ct
from torch import jit
import torch


class NNTestSingle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m = nn.Sequential(nn.Conv2d(3, 6, kernel_size=(1, 1)), nn.Conv2d(6, 3, kernel_size=(1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)


class NNTestMulti(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 6, kernel_size=(1, 1)), nn.Conv2d(6, 3, kernel_size=(1, 1)))
        self.model2 = nn.Sequential(nn.Conv2d(3, 6, kernel_size=(1, 1)), nn.Conv2d(6, 3, kernel_size=(1, 1)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x), self.model2(x)


single_out = NNTestSingle()
single_out.eval()
traced_model_single = jit.trace(single_out, example_inputs=(torch.rand(1, 3, 128, 128),))
traced_model_single(torch.rand(1, 3, 128, 128))

multi_out = NNTestMulti()
multi_out.eval()
traced_model_multi = jit.trace(multi_out, example_inputs=(torch.rand(1, 3, 128, 128),))
traced_model_multi(torch.rand(1, 3, 128, 128))


def prepare_model(traced_model):
    ct_type = ct.ImageType
    inp_kwargs = [
        {
            "name": "content",
            "shape": (1, 3, 128, 128),
        }
    ]
    inp_kwargs[0]["scale"] = 1 / 255.0
    inputs = [
        ct_type(**inp_kwargs[0]),
    ]
    model = ct.convert(
        traced_model,
        inputs=inputs,
        convert_to="neuralnetwork",
        # compute_precision=ct.precision.FLOAT32
    )
    return model


def test_enumerated_inp_img():
    shapes = [(64, 128), (128, 256), (1024, 2048)]
    inp_enum_img = InputEnumeratedShapeImage(image_shapes=shapes)
    exporter = CoreExporter([inp_enum_img])
    parsed_model = exporter(prepare_model(traced_model_single))
    assert parsed_model._spec.description.input[0].type.imageType.enumeratedSizes.sizes.__len__() > 1
    for i in range(len(shapes)):
        h, w = shapes[i]
        model_h = parsed_model._spec.description.input[0].type.imageType.enumeratedSizes.sizes[i].height

        model_w = parsed_model._spec.description.input[0].type.imageType.enumeratedSizes.sizes[i].width

        assert model_h == h and model_w == w, f"Wrong shape expect {h, w} got {model_h, model_w}"


def test_rename_output_single():
    new_name = "TestOut"
    rename = RenameOutput(new_name=new_name)
    exporter = CoreExporter([rename])
    parsed_model = exporter(prepare_model(traced_model_single))
    model_out_name = parsed_model._spec.description.output[0].name
    assert model_out_name == new_name, f"wrong name expect {new_name} got {model_out_name}"


def test_rename_output_multiple():
    new_name_one, new_name_two = "TestOut1", "TestOut2"
    chain = [
        RenameOutput(new_name=new_name_one, output_idx=0),
        RenameOutput(new_name=new_name_two, output_idx=1),
    ]
    exporter = CoreExporter(chain)
    parsed_model = exporter(prepare_model(traced_model_multi))
    model_out_name_one = parsed_model._spec.description.output[0].name
    model_out_name_two = parsed_model._spec.description.output[1].name

    assert model_out_name_one == new_name_one, f"wrong name expect {new_name_one} got {model_out_name_one}"
    assert model_out_name_two == new_name_two, f"wrong name expect {new_name_two} got {model_out_name_two}"


def test_output_static():
    chain = [OutputStaticImage(height=128, width=128, output_idx=1)]
    exporter = CoreExporter(chain)
    parsed_model = exporter(prepare_model(traced_model_multi))
    assert parsed_model._spec.description.output[1].type.imageType.height == 128
    assert parsed_model._spec.description.output[1].type.imageType.width == 128

    chain = [OutputStaticImage(height=128, width=128, output_idx=0)]
    exporter = CoreExporter(chain)
    parsed_model = exporter(prepare_model(traced_model_single))
    assert parsed_model._spec.description.output[0].type.imageType.height == 128
    assert parsed_model._spec.description.output[0].type.imageType.width == 128


def test_output_dynamic():
    chain = [
        OutputDynamicImage(
            height_lower_bound=64, height_upper_bound=2048, width_lower_bound=32, width_upper_bound=1024, output_idx=1
        )
    ]
    exporter = CoreExporter(chain)
    parsed_model = exporter(prepare_model(traced_model_multi))
    assert parsed_model._spec.description.output[1].type.imageType.imageSizeRange.heightRange.lowerBound == 64
    assert parsed_model._spec.description.output[1].type.imageType.imageSizeRange.heightRange.upperBound == 2048

    assert parsed_model._spec.description.output[1].type.imageType.imageSizeRange.widthRange.lowerBound == 32
    assert parsed_model._spec.description.output[1].type.imageType.imageSizeRange.widthRange.upperBound == 1024

    chain = [
        OutputDynamicImage(
            height_lower_bound=64, height_upper_bound=2048, width_lower_bound=32, width_upper_bound=1024, output_idx=0
        )
    ]
    exporter = CoreExporter(chain)
    parsed_model = exporter(prepare_model(traced_model_single))
    assert parsed_model._spec.description.output[0].type.imageType.imageSizeRange.heightRange.lowerBound == 64
    assert parsed_model._spec.description.output[0].type.imageType.imageSizeRange.heightRange.upperBound == 2048

    assert parsed_model._spec.description.output[0].type.imageType.imageSizeRange.widthRange.lowerBound == 32
    assert parsed_model._spec.description.output[0].type.imageType.imageSizeRange.widthRange.upperBound == 1024
