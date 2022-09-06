# HandMotion4k Dataset

This is how we make HandMotion4k Dataset, the HandMotion4k Dataset was generated from videos taken by GoPro.

## Getting Started

### Prerequisites

Requirements are listed in the requirement.txt, just simply run:

    pip install -r requirement.txt

You also need to have FFmpeg installed on your computer, click [here](https://ffmpeg.org/download.html) to install FFmpeg

## Prepare the dataset

Use the following command to extract frames from videos:

    ffmpeg -i yourvideo.MP4 -r 25 -f image2 yourvideo-%1d.png

You can use `-r` to adjust the number of frames extracted from the videos, smaller `-r` will result blurrier motion blur. 

## Generate motion-blurred image

Please create a `blur` directory and place your sharp images in `dir_sharp` directory, and run:

    python main.py

## Authors

  - **Guan Huang** -
    [54hg0220](https://github.com/54hg0220)

See also the list of
[contributors]()
who participated in this project.


