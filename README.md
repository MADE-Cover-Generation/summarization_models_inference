  Video Summarization Tool
  --------------------------
  ## Description

This tool generates video summaries using four state-of-the-art
summarization models:

-   PGL-SUM
-   CA-SUM
-   DSNet anchor based
-   DSNet anchor free

The models are pretrained on the TVSum and SumMe datasets.

# Installation

1.  **Set up a virtual environment**:

    ``` bash
    python -m venv .summarization
    ```

2.  **Activate the virtual environment**:

    ``` bash
    source .summarization/bin/activate
    ```

3.  **Install required packages**:

    ``` bash
    pip install -r requirements.txt
    ```

# Usage

## Summary Generation

1.  **Navigate to the source folder**:

    ``` bash
    cd src
    ```

2.  **Generate summary for a single video**:

    ``` bash
    python inference.py pglsum --source ../custom_data/videos/source_video_name.mp4 --save-path ./output/summary_video_name.mp4 --sample-rate 30 --final-frame-length 30
    ```

3.  **Generate summaries for a folder of videos**:

    ``` bash
    python inference.py pglsum --source ../custom_data/videos/source_video_folder --save-path ./output/summary_videos_folder --sample-rate 30 --final-frame-length 30
    ```

## Model Name References

-   `pglsum` - PGL-SUM
-   `casum` - CA-SUM
-   `dsnet_ab` - DSNet anchor based
-   `dsnet_af` - DSNet anchor free

## Parameters Explanation

-   `--sample-rate 30`: The model analyzes every 30th frame.
-   `--final-frame-length 30`: The resulting video summary will contain
    around 30 frames, roughly equivalent to 27 seconds. This duration
    can vary between 23-31 seconds depending on the frames per second of
    the original video.
-   `--max-shot-length 8`: A single shot in the summary won\'t exceed 8
    frames.
-   `--min-penalty-shot-length 5`: Shots that are 5 frames or shorter
    will incur a length penalty, thus making them less likely to appear
    in the final summary.
