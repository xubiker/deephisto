# DeepHisto

Repository, contatining a set of methods, tools and models for histological image analysis developed at [MMIP lab](http://imaging.cs.msu.ru).

This readme is not complete yet. Sorry...

## Installation

(Not ready)

Create env:
```conda env create -n deephisto -f environment.yml```

## Run examples

The examples of using different methods can be found in [examples folder](/examples/).

Use  patch samplers to extract patches from annotated regions. Can be used for training classification models.
`python -m examples.sample_annotated_dense`
`python -m examples.sample_annotated_rnd`
`python -m examples.sample_annotated_rnd --torch`

Use patch samplers to extract patches from whole image. Can be random or dense. Is usefull for predicting for the whole image.
`python -m examples.sample_full_dense`
`python -m examples.sample_full_random`

Train simlpe patch-based model:
`python -m models.patch_cls_simple.train`
`python -m models.patch_cls_simple.train --extract_test`

Make full prediction on WSI image using saved model:
`python -m examples.predict_full_patched`
