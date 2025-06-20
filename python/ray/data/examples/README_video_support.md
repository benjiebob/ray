# Video Support in Ray Data LLM

This document provides an overview of the video support feature added to `ray.data.llm`, which allows processing videos with vision language models (VLMs) like Qwen2.5-VL and InternVL3.

## Overview

Ray Data LLM now supports video processing for batch inference with VLMs. This feature enables applications to analyze video content and answer questions about videos using large multimodal models.

## Implementation Details

The implementation adds video support to Ray Data LLM through the following components:

1. **Configuration**: Added `has_video` parameter to `OfflineProcessorConfig` to indicate that the input data contains videos.

2. **Video Processing Stage**: Created a new `PrepareVideoStage` that:
   - Extracts video data from messages
   - Downloads videos from URLs if needed
   - Processes videos using OpenCV
   - Extracts frames at specified fps or total frame count
   - Converts video frames to the format expected by VLMs

3. **Processor Integration**: Updated the `vLLMEngineProcessor` to include the video stage when `has_video=True` is set.

4. **Parameter Configuration**: Added support for video processing parameters like fps and frame count through the `mm_processor_kwargs` in the engine configuration.

## Usage

To use video processing in your application:

1. Set `has_video=True` in the `vLLMEngineProcessorConfig`
2. Configure video parameters in `mm_processor_kwargs` (fps, num_frames, etc.)
3. In your preprocessor function, include video content in the right format based on the model's requirements

### Basic Example

```python
import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

# Create dataset with videos
videos_ds = ray.data.from_items([
    {"video_path": "path/to/video.mp4", "question": "What's in this video?"}
])

# Configure processor with video support
config = vLLMEngineProcessorConfig(
    model_source="Qwen/Qwen2.5-VL-3B-Instruct",
    engine_kwargs=dict(
        mm_processor_kwargs={"fps": 1},  # Extract 1 frame per second
    ),
    has_video=True,  # Enable video processing
    batch_size=16,
)

# Define preprocessor that includes video
def preprocess(row):
    return dict(
        messages=[
            {"role": "user", 
             "content": [
                {"type": "text", "text": row["question"]},
                {"type": "video", "video": row["video_path"]}
             ]}
        ],
    )

# Build and use processor
processor = build_llm_processor(config, preprocess=preprocess)
results = processor(videos_ds)
```

### Full Example

For a complete working example, see the provided script `python/ray/data/examples/video_processing_example.py` which demonstrates how to use both Qwen2.5-VL and InternVL3 for video processing.

To run the example:

```bash
python -m ray.data.examples.video_processing_example --video_path path/to/video.mp4 --model qwen
```

## Supported Models

The video support has been tested with:

- **Qwen2.5-VL**: Uses `<|video_pad|>` as the placeholder and video content in the user message
- **InternVL3**: Uses `<video>` as the placeholder and video content in the user message

Other models that support video processing through vLLM should also work by following their specific prompt formats.

## Configuration Options

The video processing can be configured with these parameters:

- `fps`: Frames per second to extract (default: 1)
- `num_frames`: Total number of frames to extract evenly spaced (default: -1, meaning use fps-based extraction)
- `min_pixels` and `max_pixels`: Control frame size limitations

These are specified in the `mm_processor_kwargs` dictionary in the engine configuration.

## Requirements

- OpenCV (cv2) for video processing
- PIL for image processing
- vLLM with version 0.7.2 or higher
