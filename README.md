# INF573 Final Project - Synthetic defocus for portrait photos

This projects aims to synthesize defocused effect for portrait images. It makes use of a Mask R-CNN model with image processing techniques.

## Prerequisites

Make sure you have all required packages

```
pip install -r requirements.txt
```

## Running the tests

Change directory to the project folder

```
cd portrait-mode-main
```

To run a demo on a single-image input, you can simply execute the script `main.py`

```
python main.py
```

To run a demo on a image-folder input, specify the `--input_path` argument

```
python main.py --input_path="sample_images" [--gpu=bool]
```

In the end of the process, you should find all the ouput images in folder `output` in the same level of the project folder.

## Execute on your own input

Execute the script with 2 optional arguments:

	*`--input_path`: (String) Path to your input, either an image file or a folder of images
	*`--gpu`: (Boolean) Whether to use GPU or not.
