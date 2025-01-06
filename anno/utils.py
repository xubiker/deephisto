"""
In future this module should be updated with petroscope features
and moved to the psanno package.
"""

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import distinctipy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw


@dataclass
class AnnoClass:
    """Class that stores each annotation class with id, color,
    description, etc."""

    id: int
    label: str
    alternate_labels: tuple[str] = ()
    description: str = None
    color: tuple[int, int, int] = None

    def __str__(self) -> str:
        label = self.label
        if self.alternate_labels:
            label += " (" + ", ".join(self.alternate_labels) + ")"
        description = ", " + self.description if self.description else ""
        return f"AnnoClass [{self.id}, {label}, {self.color}{description}]"

    @property
    def label_full(self) -> str:
        """Return string containing main label with all alternative labels."""
        if not self.alternate_labels:
            return self.label
        return self.label + " (" + ", ".join(self.alternate_labels) + ")"


class AnnoDescription:
    """Class storing the set of annotation classes."""

    def __init__(self, _anno_classes: tuple[AnnoClass]) -> None:
        self.anno_classes = _anno_classes
        self.anno_classes_dict = self._build_anno_dict(_anno_classes)

    def _build_anno_dict(
        self, anno_classes: tuple[AnnoClass]
    ) -> dict[str, AnnoClass]:
        anno_dict = {c.label: c for c in anno_classes}
        for cls in anno_classes:
            if cls.alternate_labels:
                anno_dict.update(
                    {alt_label: cls for alt_label in cls.alternate_labels}
                )
        return anno_dict

    @classmethod
    def with_known_colors(
        cls, labels_with_color: dict[str, (int, int, int)]
    ) -> "AnnoDescription":
        """creates AnnoDescription with known colors.

        Args:
            labels_with_color (dict[str,): labels and corresponding colors

        Returns:
            AnnoDescription: AnnoDescription object
        """
        anno_classes = [
            AnnoClass(id=i, label=lbl, color=color)
            for i, (lbl, color) in enumerate(labels_with_color.items())
        ]
        return AnnoDescription(anno_classes)

    @classmethod
    def with_auto_colors(cls, labels: Iterable[str]) -> "AnnoDescription":
        """creates AnnoDescription with automatically generated colors.

        Args:
            labels (Iterable[str]): labels that are going to be used.

        Returns:
            AnnoDescription: AnnoDescription object
        """
        palette = Palette(n_colors_max=len(labels), rng=42)
        anno_classes = [
            AnnoClass(id=i, label=lbl, color=palette.colors[i])
            for i, lbl in enumerate(labels)
        ]
        return AnnoDescription(anno_classes)

    @classmethod
    def auto_from_files(cls, path: list[Path] | Path) -> "AnnoDescription":
        """creates AnnoDescription from annotation file or folder with files.

        Args:
            path (list[Path] | Path): path to annotation file or
            folder with annotation files.

        Raises:
            PSImageException: if no annotation files found

        Returns:
            AnnoDescription: AnnoDescription object
        """
        print("Extracting class labels from annotation files...")
        anno_files = []
        if path.is_dir():
            anno_files = [f for f in path.iterdir() if f.suffix == ".json"]
        elif path.is_file() and path.suffix == ".json":
            anno_files = [path]
        if not anno_files:
            raise RuntimeError("No annotation files found")

        labels = set()
        for f in anno_files:
            with f.open("r") as anno_json:
                for anno in json.load(anno_json):
                    if isinstance(anno, dict):
                        labels.add(anno["class"])
        labels = sorted(list(labels))
        print(f"{len(labels)} labels {labels} found.")
        return cls.with_auto_colors(labels)

    def color_by_label(self, label: str) -> tuple[int, int, int]:
        """Return color of the given label.

        Args:
            label (str): label

        Returns:
            (int, int, int): color
        """
        return self.anno_classes_dict[label].color


class Palette:
    """Class for generating distinct colors."""

    def __init__(
        self,
        colors: tuple[(int, int, int)] = None,
        n_colors_max: int = None,
        rng: int = None,
    ) -> None:
        """Generate palette of distinct colors.

        Args:
            colors (tuple[, optional): Known colors. Defaults to None.

            n_colors_max (int, optional): Number of maximum possible colors.
            Defaults to None.

            rng (int, optional): seed for random number generator.
            Defaults to None.

        Raises:
            PSImageException: for incorrect arguments
        """
        if colors is not None:
            for color in colors:
                if not all(0 <= c <= 255 for c in color):
                    raise RuntimeError(
                        "Color values must be between 0 and 255."
                    )
        if n_colors_max is not None:
            if colors is not None and n_colors_max < len(colors):
                raise RuntimeError(
                    "n_colors_max must be >= number of defined colors."
                )
        n_colors_max = len(colors) if n_colors_max is None else n_colors_max
        self.colors = list(colors) if colors is not None else []

        additional_colors = distinctipy.get_colors(
            n_colors_max - len(self.colors),
            [(cl[0] / 255, cl[1] / 255, cl[2] / 255) for cl in self.colors],
            pastel_factor=0.1,
            rng=rng,
        )
        additional_colors = [
            (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            for color in additional_colors
        ]
        self.colors.extend(additional_colors)


@dataclass
class AnnoVisualizerParams:
    """Dataclass that stores parameters of annotations visualization."""

    fill: bool
    fill_transparency: float
    line_width: int
    show_legend: bool
    legend_placement: str
    legend_size: int

    @classmethod
    def default(cls) -> "AnnoVisualizerParams":
        """create default AnnoVisualizerParams."""
        return AnnoVisualizerParams(
            fill=True,
            fill_transparency=0.3,
            line_width=2,
            show_legend=True,
            legend_placement="TR",
            legend_size=20,
        )

    @classmethod
    def no_legend(cls) -> "AnnoVisualizerParams":
        """create default AnnoVisualizerParams."""
        return AnnoVisualizerParams(
            fill=True,
            fill_transparency=0.3,
            line_width=2,
            show_legend=False,
            legend_placement=None,
            legend_size=None,
        )


@dataclass
class PatchVisAccent:
    layer: int
    size: int
    x: int
    y: int
    label: str = None

    @classmethod
    def parse(
        cls, code_str: str, layer: int, patch_s: int
    ) -> "PatchVisAccent":
        # "r28_LP_7_x17311_y14066"
        s = code_str.split("_")
        label = s[1]
        x = int(s[3][1:])
        y = int(s[4][1:])
        return PatchVisAccent(layer=layer, size=patch_s, x=x, y=y, label=label)


class AnnoVisualizer:
    """Class for visualizing annotations on images."""

    def __init__(
        self,
        anno_description: AnnoDescription,
        vis_params: AnnoVisualizerParams = AnnoVisualizerParams.default(),
    ) -> None:
        self.anno_description = anno_description
        self.vis_params = vis_params

    def visualize(
        self,
        psimage,
        polygon_annotations: list[(str, np.array)],
        scale: float = None,
        max_side: int = None,
        auto_downscale=False,
        patch_accents: list[(str, PatchVisAccent)] = None,
    ) -> Image:
        """Create image preview with drawn polygonal annotations.

        Args:
            psimage (_type_): PSImage object

            polygon_annotations (list[): polygon annotations as
            list of pairs <label, vertices>

            scale (float | None, optional): downscale coefficient (< 1).
            If None, max_size parameter is used. Defaults to None.

            max_side (int | None, optional): maximum side size
            (width or height) of the exported image.
            If None, scale parameter is used.

            auto_downscale (bool, optional): Automatically downscale the image
            to max_size_limit if image size is too big.

        Returns:
            Image: PIL Image with drawn annotations.

        Throws:
            PSImageException: if image is too big and auto_downscale is False.
        """
        vp = self.vis_params

        img = psimage.to_image(
            max_side=max_side, scale=scale, auto_downscale=auto_downscale
        )

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        downscale_factor = (
            img.height / psimage.height + img.width / psimage.width
        ) / 2

        fill_transparency = int(255 * vp.fill_transparency) if vp.fill else 0

        for polygon_anno in polygon_annotations:
            lbl, poly = polygon_anno
            color = self.anno_description.color_by_label(lbl)

            vertices_downscaled = [
                (x * downscale_factor, y * downscale_factor) for x, y in poly
            ]
            draw.polygon(
                vertices_downscaled,
                outline=color + (255,),
                width=vp.line_width,
                fill=color + (fill_transparency,),
            )

        if patch_accents is not None:
            self._add_patch_accents(draw, downscale_factor, patch_accents)

        img_final = Image.alpha_composite(
            img.convert("RGBA"), overlay
        ).convert("RGB")

        if self.vis_params.show_legend:
            img_final = self._add_legend(img_final)

        return img_final.convert("RGB")

    def _add_patch_accents(
        self,
        draw: ImageDraw,
        downscale_factor: float,
        patch_accent: list[PatchVisAccent],
    ):
        vp = self.vis_params
        fill_transparency = int(255 * vp.fill_transparency) if vp.fill else 0
        fill_transparency = min(255, fill_transparency + 80)

        for pa in patch_accent:
            color = self.anno_description.color_by_label(pa.label)
            # shift color
            color = (
                min(255, color[0] + 20),
                max(0, color[1] - 10),
                min(255, color[2] + 10),
            )

            x, y = (
                pa.layer * pa.x * downscale_factor,
                pa.layer * pa.y * downscale_factor,
            )
            s = pa.layer * pa.size * downscale_factor
            v = [
                (x, y),
                (x + s, y),
                (x + s, y + s),
                (x, y + s),
            ]
            draw.polygon(
                v,
                outline=color + (255,),
                width=1,
                fill=color + (fill_transparency,),
            )

    def _add_legend(self, img: Image, dpi=100) -> Image:
        fig = plt.figure(
            figsize=(img.width / dpi, img.height / dpi),
        )
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        plt.imshow(img)
        legend_data = [
            (i.color, i.label_full) for i in self.anno_description.anno_classes
        ]

        handles = [
            Rectangle((0, 0), 1, 1, color=[v / 255 for v in c])
            for c, _ in legend_data
        ]
        labels = [lbl for _, lbl in legend_data]

        legend_loc = {
            "TL": "upper left",
            "TR": "upper right",
            "BR": "lower right",
            "BL": "lower left",
        }[self.vis_params.legend_placement]

        plt.legend(
            handles,
            labels,
            loc=legend_loc,
            prop={"size": self.vis_params.legend_size},
        )

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        im = Image.open(buf).copy()
        buf.close()

        return im
