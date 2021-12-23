from dataclasses import dataclass
from typing import Any, Union
import coremltools as ct
from abc import ABC, abstractmethod
from coremltools.proto import FeatureTypes_pb2 as ft
from coremltools.models.neural_network import flexible_shape_utils


class AbstractUtil(ABC):
    pass


@dataclass
class OutputStaticImage(AbstractUtil):
    """
    Conver coreml output to image with static shape
    """

    height: int
    width: int
    output_idx: int = 0
    inplace: bool = True
    colorspace: str = "RGB"
    model_type = ct.models.MLModel

    def __call__(self, model: ct.models.MLModel) -> None:
        model._spec.description.output[self.output_idx].type.imageType.width = self.width
        model._spec.description.output[self.output_idx].type.imageType.height = self.height
        model._spec.description.output[
            self.output_idx
        ].type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value(self.colorspace)


@dataclass
class OutputDynamicImage(AbstractUtil):
    """
    Convert coreml output to image with dynamic shape
    """

    height_lower_bound: int
    height_upper_bound: int
    width_lower_bound: int
    width_upper_bound: int
    output_idx: int = 0
    inplace: bool = True
    colorspace: str = "RGB"
    model_type = ct.models.MLModel

    def __call__(self, model) -> None:
        model._spec.description.output[
            self.output_idx
        ].type.imageType.imageSizeRange.heightRange.lowerBound = self.height_lower_bound
        model._spec.description.output[
            self.output_idx
        ].type.imageType.imageSizeRange.widthRange.lowerBound = self.width_lower_bound

        model._spec.description.output[
            self.output_idx
        ].type.imageType.imageSizeRange.heightRange.upperBound = self.height_upper_bound
        model._spec.description.output[
            self.output_idx
        ].type.imageType.imageSizeRange.widthRange.upperBound = self.width_upper_bound

        model._spec.description.output[
            self.output_idx
        ].type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value(self.colorspace)


@dataclass
class RenameOutput(AbstractUtil):
    """
    Rename output names for coreml models (as coreml dosent work with pytorch output naming etc.)
    """

    new_name: str
    output_idx: int = 0
    inplace = True
    model_type = ct.models.MLModel

    def __call__(self, model: ct.models.MLModel) -> None:
        ct.utils.rename_feature(
            model._spec,
            model._spec.description.output[self.output_idx].name,
            self.new_name,
            rename_outputs=True,
        )


@dataclass
class InputEnumeratedShapeImage(AbstractUtil):
    """Change input using enumertated shapes (best performance with some flexibility left)
    coremldocs -> https://coremltools.readme.io/docs/flexible-inputs#select-from-predetermined-shapes
    """

    image_shapes: list[tuple[int, int]]  # example -> [(H, W), (H, W)]
    input_idx: int = 0
    inplace: bool = True
    colorspace: str = "RGB"
    model_type = ct.models.MLModel

    def __post_init__(self):
        assert all(
            [len(i) == 2 for i in self.image_shapes]
        ), f"All image shapes should have only 2 dimmensions height and width got {([len(i) for i in self.image_shapes if len(i) != 2])}"

    def __call__(self, model: ct.converters.mil.Program) -> None:
        input_name = model._spec.description.input[self.input_idx].name

        flexible_shape_utils.add_enumerated_image_sizes(
            model._spec,
            input_name,
            [flexible_shape_utils.NeuralNetworkImageSize(i[0], i[1]) for i in self.image_shapes],
        )


@dataclass
class CoreExporter:
    """Main CoreExporter class take all
    Example:
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
        # Old model output name => var27, Old model output type Array [Avaliable input shapes 128x128]

    Returns:
        Union[ct.converters.mil.Program, ct.models.MLModel]:
    """

    utils_methods: list[AbstractUtil]

    def __call__(
        self, model: Union[ct.converters.mil.Program, ct.models.MLModel]
    ) -> Union[ct.converters.mil.Program, ct.models.MLModel]:

        assert all(
            [util.model_type == type(model) for util in self.utils_methods]
        ), f"Wrong model type util used expected => {type(model)} for CoreExporter chain"

        for util in self.utils_methods:
            if util.inplace:
                util(model)
            else:
                model = util(model)
        return model
