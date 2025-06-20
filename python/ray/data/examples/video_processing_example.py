"""
Example script for video processing with ray.data.llm.

This example demonstrates how to use ray.data.llm to process videos
with vision language models like Qwen2.5-VL and InternVL3.
"""

import argparse
import os
import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor


def process_videos_with_qwen(videos_dataset, hf_token=None):
    """Process videos with Qwen2.5-VL."""
    print("Processing videos with Qwen2.5-VL...")
    
    # Configure the processor for Qwen2.5-VL
    config = vLLMEngineProcessorConfig(
        model_source="Qwen/Qwen2.5-VL-3B-Instruct",
        engine_kwargs=dict(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            max_model_len=4096,
            enable_chunked_prefill=True,
            max_num_batched_tokens=2048,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "fps": 1,  # Extract frames at 1 fps
            },
        ),
        runtime_env=dict(
            env_vars=dict(
                HF_TOKEN=hf_token,
            ),
        ) if hf_token else None,
        batch_size=16,
        accelerator_type="L4",  # Adjust based on hardware
        concurrency=1,
        has_video=True,  # Enable video processing
    )
    
    def preprocess(row):
        return dict(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can analyze videos."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": row["question"]
                        },
                        {
                            "type": "video",
                            "video": row["video_path"]
                        }
                    ]
                },
            ],
            sampling_params=dict(
                temperature=0.3,
                max_tokens=150,
            ),
        )
    
    def postprocess(row):
        return {
            "response": row["generated_text"],
        }
    
    processor = build_llm_processor(
        config,
        preprocess=preprocess,
        postprocess=postprocess,
    )
    
    processed_videos = processor(videos_dataset).materialize()
    return processed_videos


def process_videos_with_internvl(videos_dataset, hf_token=None):
    """Process videos with InternVL3."""
    print("Processing videos with InternVL3...")
    
    # Configure the processor for InternVL3
    config = vLLMEngineProcessorConfig(
        model_source="OpenGVLab/InternVL3-2B",
        engine_kwargs=dict(
            trust_remote_code=True,
            max_model_len=8192,
            mm_processor_kwargs={
                "fps": 1,  # Extract frames at 1 fps
            },
        ),
        runtime_env=dict(
            env_vars=dict(
                HF_TOKEN=hf_token,
            ),
        ) if hf_token else None,
        batch_size=16,
        accelerator_type="L4",  # Adjust based on hardware
        concurrency=1,
        has_video=True,  # Enable video processing
    )
    
    def preprocess(row):
        return dict(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": row["video_path"]
                        },
                        {
                            "type": "text",
                            "text": row["question"]
                        }
                    ]
                }
            ],
            sampling_params=dict(
                temperature=0.3,
                max_tokens=150,
            ),
        )
    
    def postprocess(row):
        return {
            "response": row["generated_text"],
        }
    
    processor = build_llm_processor(
        config,
        preprocess=preprocess,
        postprocess=postprocess,
    )
    
    processed_videos = processor(videos_dataset).materialize()
    return processed_videos


def main(args):
    """Main function."""
    ray.init(ignore_reinit_error=True)
    
    # Create a dataset with video paths
    videos_dataset = ray.data.from_items([
        {"video_path": args.video_path, 
         "question": "What's happening in this video?"}
    ])
    
    # Process with the selected model
    if args.model == "qwen":
        results = process_videos_with_qwen(videos_dataset, args.hf_token)
    elif args.model == "internvl":
        results = process_videos_with_internvl(videos_dataset, args.hf_token)
    else:
        print(f"Unknown model: {args.model}")
        return
    
    # Show results
    print("\nResults:")
    results.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos with VLMs")
    parser.add_argument("--video_path", type=str, required=True, 
                        help="Path to the video file")
    parser.add_argument("--model", type=str, choices=["qwen", "internvl"], default="qwen", 
                        help="VLM model to use")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"), 
                        help="HuggingFace token (if needed)")
    
    args = parser.parse_args()
    main(args)
