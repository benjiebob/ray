"""Prepare Video Stage"""
import asyncio
import base64
import importlib
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)
from urllib.parse import urlparse

import aiohttp
import numpy as np
import requests

from ray.llm._internal.batch.stages.base import (
    StatefulStage,
    StatefulStageUDF,
)
from ray.llm._internal.batch.stages.prepare_image_stage import HTTPConnection

# TODO: Remove the guard once OpenCV is added into the dependencies.
if TYPE_CHECKING:
    import cv2
    from PIL import Image

logger = logging.getLogger(__name__)

_VideoType = Union[str, bytes, np.ndarray, List[np.ndarray], List["Image.Image"]]


class VideoProcessor:
    """Process videos for vision language models."""

    def __init__(self):
        """Initialize the video processor."""
        self.cv2 = importlib.import_module("cv2")
        self.Image = importlib.import_module("PIL.Image")
        self.http_connection = HTTPConnection()

    async def download_video_from_url(self, video_url: str) -> Optional[bytes]:
        """Download the video from the Internet with up to 3 retries.

        Args:
            video_url: The video URL to download.

        Returns:
            The video bytes (None if failed to download).
        """
        for _ in range(3):
            try:
                video_raw = await self.http_connection.async_get_bytes(
                    video_url, timeout=10
                )
                return video_raw
            except Exception:
                await asyncio.sleep(1)
        return None

    async def load_video_bytes_from_url(self, video_urls: List[str]) -> List[bytes]:
        """Load videos from URLs.

        Args:
            video_urls: The video URLs to load.

        Returns:
            The video bytes.
        """
        return await asyncio.gather(
            *[self.download_video_from_url(video_url) for video_url in video_urls]
        )

    def video_to_ndarrays(
        self, video_data: bytes, num_frames: int = -1, fps: int = 1
    ) -> np.ndarray:
        """Convert video data to numpy arrays.

        Args:
            video_data: The video data in bytes.
            num_frames: Number of frames to extract. If -1, extract all frames.
            fps: Frames per second to extract. Only used if num_frames == -1.

        Returns:
            A numpy array of frames.
        """
        # Write temporary file to disk
        temp_file = Path("temp_video.mp4")
        try:
            with open(temp_file, "wb") as f:
                f.write(video_data)
            
            cap = self.cv2.VideoCapture(str(temp_file))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file")

            total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError(f"Could not determine frame count")
                
            # Determine frame extraction strategy
            frames = []
            
            if num_frames > 0:
                # Extract specific number of frames evenly distributed
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                for idx in range(total_frames):
                    ok = cap.grab()  # next img
                    if not ok:
                        break
                    if idx in frame_indices:  # only decompress needed
                        ret, frame = cap.retrieve()
                        if ret:
                            frames.append(frame)
            else:
                # Extract frames at specific fps
                video_fps = int(cap.get(self.cv2.CAP_PROP_FPS))
                if video_fps <= 0:
                    video_fps = 30  # Default assumption if FPS can't be determined
                    
                frame_interval = max(1, int(video_fps / fps))
                for idx in range(total_frames):
                    ok = cap.grab()
                    if not ok:
                        break
                    if idx % frame_interval == 0:
                        ret, frame = cap.retrieve()
                        if ret:
                            frames.append(frame)
            
            cap.release()
            
            if not frames:
                raise ValueError(f"Could not extract any frames from video")
                
            return np.stack(frames)
        finally:
            # Clean up the temporary file
            if temp_file.exists():
                temp_file.unlink()

    def video_to_pil_images_list(
        self, video_data: bytes, num_frames: int = -1, fps: int = 1
    ) -> List["Image.Image"]:
        """Convert video data to a list of PIL Images.

        Args:
            video_data: The video data in bytes.
            num_frames: Number of frames to extract. If -1, extract frames based on fps.
            fps: Frames per second to extract if num_frames is -1.

        Returns:
            A list of PIL Images.
        """
        frames = self.video_to_ndarrays(video_data, num_frames, fps)
        return [
            self.Image.fromarray(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
            for frame in frames
        ]

    async def fetch_videos_from_urls(
        self, video_urls: List[str], num_frames: int = -1, fps: int = 1
    ) -> List[List["Image.Image"]]:
        """Load videos from URLs and convert to PIL Images.

        Args:
            video_urls: URLs of the videos.
            num_frames: Number of frames to extract per video.
            fps: Frames per second to extract if num_frames is -1.

        Returns:
            A list of lists of PIL Images (one list per video).
        """
        video_bytes = await self.load_video_bytes_from_url(video_urls)
        return [
            self.video_to_pil_images_list(vb, num_frames, fps) 
            for vb in video_bytes if vb is not None
        ]

    def _load_video_from_data_url(
        self, video_url: str, num_frames: int = -1, fps: int = 1
    ) -> List["Image.Image"]:
        """Load a video from a base64 data URL.

        Args:
            video_url: A data URL containing base64-encoded video data.
            num_frames: Number of frames to extract.
            fps: Frames per second to extract if num_frames is -1.

        Returns:
            A list of PIL Images.
        """
        # Only split once and assume the second part is the base64 encoded video
        _, video_base64 = video_url.split(",", 1)
        video_data = base64.b64decode(video_base64)
        return self.video_to_pil_images_list(video_data, num_frames, fps)

    async def process(
        self, 
        videos: List[_VideoType], 
        num_frames: int = -1, 
        fps: int = 1
    ) -> List[List["Image.Image"]]:
        """Process videos into frames suitable for vision language models.

        Args:
            videos: A list of videos (URLs, base64 data URLs, or already loaded data).
            num_frames: Number of frames to extract per video.
            fps: Frames per second to extract if num_frames is -1.

        Returns:
            A list of lists of PIL Images (one list per video).
        """
        if not videos:
            return []

        # Check the type of the first video to determine processing approach
        if isinstance(videos[0], str):
            # Check if it's a URL or a data URL
            if videos[0].startswith("http"):
                return await self.fetch_videos_from_urls(videos, num_frames, fps)
            elif videos[0].startswith("data:video"):
                return [self._load_video_from_data_url(video, num_frames, fps) for video in videos]
            else:
                raise ValueError(f"Invalid video URL prefix: {videos[0]}")
        elif isinstance(videos[0], bytes):
            return [self.video_to_pil_images_list(video, num_frames, fps) for video in videos]
        elif isinstance(videos[0], np.ndarray):
            # Assuming it's already a single frame or multiple frames
            if len(videos[0].shape) == 3:  # Single frame
                frames_list = [[self.Image.fromarray(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))] 
                              for frame in videos]
                return frames_list
            elif len(videos[0].shape) == 4:  # Multiple frames
                frames_list = [[self.Image.fromarray(self.cv2.cvtColor(f, self.cv2.COLOR_BGR2RGB)) 
                               for f in video] for video in videos]
                return frames_list
        elif isinstance(videos[0], list):
            # Already a list of frames (either numpy arrays or PIL Images)
            if isinstance(videos[0][0], np.ndarray):
                frames_list = [[self.Image.fromarray(self.cv2.cvtColor(f, self.cv2.COLOR_BGR2RGB)) 
                              for f in video] for video in videos]
                return frames_list
            elif isinstance(videos[0][0], self.Image.Image):
                return videos  # Already in the right format
        
        raise ValueError(f"Unsupported video type: {type(videos[0])}")


class PrepareVideoUDF(StatefulStageUDF):
    def __init__(
        self,
        data_column: str,
        expected_input_keys: List[str],
        num_frames: int = -1,
        fps: int = 1,
    ):
        """Initialize the video preparation UDF.

        Args:
            data_column: The data column name.
            expected_input_keys: The expected input keys.
            num_frames: Number of frames to extract per video.
            fps: Frames per second to extract if num_frames is -1.
        """
        super().__init__(data_column, expected_input_keys)
        self.num_frames = num_frames
        self.fps = fps
        self.Image = importlib.import_module("PIL.Image")
        self.video_processor = VideoProcessor()

    def extract_video_info(self, messages: List[Dict]) -> List[_VideoType]:
        """Extract video information from chat messages.

        Args:
            messages: List of chat messages.

        Returns:
            List of video information.
        """
        video_info: List[_VideoType] = []
        for message in messages:
            if not isinstance(message["content"], list):
                continue
            for content in message["content"]:
                if content["type"] not in ("video", "video_url"):
                    continue
                video = content[content["type"]]
                if not isinstance(video, str) and not isinstance(
                    video, (bytes, np.ndarray, list)
                ):
                    raise ValueError(f"Cannot handle video type {type(video)}")
                video_info.append(video)
        return video_info

    async def udf(self, batch: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        messages = [row["messages"] for row in batch]

        # Process all videos in this batch
        all_video_info = [self.extract_video_info(message) for message in messages]
        flat_all_video_info = [video for videos in all_video_info for video in videos]
        
        flat_all_frames = await self.video_processor.process(
            flat_all_video_info, self.num_frames, self.fps
        )

        # Map the processed frames back to the corresponding messages
        video_start_idx = 0
        idx_in_batch = 0
        for video_info_per_req in all_video_info:
            num_videos_in_req = len(video_info_per_req)
            ret = {self.IDX_IN_BATCH_COLUMN: idx_in_batch}
            idx_in_batch += 1
            if num_videos_in_req > 0:
                frames_lists = flat_all_frames[
                    video_start_idx : video_start_idx + num_videos_in_req
                ]
                # Flatten all frames from all videos in this request
                all_frames = [frame for frames_list in frames_lists for frame in frames_list]
                ret.update(
                    {
                        "video_frames": all_frames,
                        "video_frames_count": [len(frames) for frames in frames_lists],
                        "frame_sizes": [(img.width, img.height) for img in all_frames],
                    }
                )
                video_start_idx += num_videos_in_req
            yield ret


class PrepareVideoStage(StatefulStage):
    """A stage to prepare videos from OpenAI chat template messages."""

    fn: StatefulStageUDF = PrepareVideoUDF

    def __init__(
        self,
        fn_constructor_kwargs: Optional[Dict] = None,
        map_batches_kwargs: Optional[Dict] = None,
        num_frames: int = -1,
        fps: int = 1,
    ):
        """Initialize the video preparation stage.

        Args:
            fn_constructor_kwargs: The kwargs to pass to the constructor of the UDF.
            map_batches_kwargs: The kwargs to pass to map_batches.
            num_frames: Number of frames to extract per video.
            fps: Frames per second to extract if num_frames is -1.
        """
        self.num_frames = num_frames
        self.fps = fps
        super().__init__(fn_constructor_kwargs, map_batches_kwargs)

    def get_fn_constructor_kwargs(self, data_column: str, expected_input_keys: List[str]) -> Dict:
        kwargs = super().get_fn_constructor_kwargs(data_column, expected_input_keys)
        kwargs["num_frames"] = self.num_frames
        kwargs["fps"] = self.fps
        return kwargs

    def get_required_input_keys(self) -> Dict[str, str]:
        """The required input keys of the stage and their descriptions."""
        return {
            "messages": "A list of messages in OpenAI chat format. "
            "See https://platform.openai.com/docs/api-reference/chat/create "
            "for details. For video, the message content should include "
            "an entry with 'type': 'video' or 'video_url'."
        }
