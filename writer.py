# Standard library imports
import io
import json
import re
import tarfile
from pathlib import Path

# Third-party imports
import numpy as np
from PIL import Image


class WebDatasetWriter:
    """
    A class for writing machine learning datasets in WebDataset format.
    
    WebDataset is a format that stores samples in TAR archives, where each sample
    consists of multiple files (e.g., image, depth, segmentation, metadata) that
    share a common key/index.
    
    This writer is designed for robotics/computer vision datasets containing:
    - RGB images
    - Depth maps
    - Segmentation maps
    - Camera poses and intrinsics
    - Joint angles
    - Instance attribute maps
    - Metadata
    """
    
    def __init__(self, output_dir: str, samples_per_shard: int = 1000):
        """
        Initialize the WebDataset writer.
        
        Args:
            output_dir: Directory where TAR shards will be written
            samples_per_shard: Number of samples to include in each TAR shard
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_per_shard = samples_per_shard
        
        # Counters for tracking progress
        self.sample_counter = 0  # Global sample counter across all shards
        self.shard_counter = self._find_last_shard_index() + 1  # Current shard index
        self.shard_sample_count = 0  # Samples in current shard
        
        # Current TAR file handle
        self.tar = None
        
        # Start writing the first shard
        self._start_new_shard()

    def _find_last_shard_index(self):
        """
        Find the highest shard index in the output directory to enable resuming.
        
        Returns:
            int: The highest existing shard index, or -1 if no shards exist
        """
        pattern = re.compile(r"dataset_shard_(\d{6})\.tar")
        shards = list(self.output_dir.glob("dataset_shard_*.tar"))
        max_idx = -1
        
        for shard in shards:
            match = pattern.match(shard.name)
            if match:
                idx = int(match.group(1))
                if idx > max_idx:
                    max_idx = idx
        
        return max_idx

    def _start_new_shard(self):
        """
        Close the current shard (if any) and start a new TAR file.
        """
        # Close previous shard if it exists
        if self.tar:
            self.tar.close()
        
        # Create new shard with zero-padded 6-digit index
        shard_name = self.output_dir / f"dataset_shard_{self.shard_counter:06d}.tar"
        self.tar = tarfile.open(shard_name, "w")
        self.shard_sample_count = 0
        self.shard_counter += 1
        
        print(f"Started new shard: {shard_name}")

    def _write_bytes(self, name: str, data: bytes):
        """
        Write raw bytes to the current TAR archive.
        
        Args:
            name: Filename within the TAR archive
            data: Raw bytes to write
        """
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        self.tar.addfile(tarinfo=info, fileobj=io.BytesIO(data))

    def _add_numpy(self, key: str, arr: np.ndarray):
        """
        Add a numpy array to the TAR archive in .npy format.
        
        Args:
            key: Filename (with extension) for the array in the archive
            arr: Numpy array to save
        """
        buffer = io.BytesIO()
        np.save(buffer, arr)
        self._write_bytes(key, buffer.getvalue())

    def _add_rgb_image(self, key: str, arr: np.ndarray):
        """
        Add an RGB image to the TAR archive as a PNG file.
        
        Args:
            key: Filename (with extension) for the image in the archive
            arr: RGB image array, either float [0,1] or uint8 [0,255]
        """
        # Convert float images to uint8 range
        if arr.dtype in [np.float32, np.float64]:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        
        # Create PIL Image and save as PNG
        img = Image.fromarray(arr)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        self._write_bytes(key, buffer.getvalue())

    def _add_json(self, key: str, meta: dict):
        """
        Add a dictionary as a JSON file to the TAR archive.
        
        Args:
            key: Filename (with extension) for the JSON file in the archive
            meta: Dictionary to serialize as JSON
        """
        data = json.dumps(meta).encode('utf-8')
        self._write_bytes(key, data)

    def add_sample(self, sample: dict):
        """
        Add a complete sample to the dataset.
        
        Each sample should contain the following keys:
        - rgb: RGB image array (H, W, 3)
        - depth: Depth map array (H, W)
        - seg: Segmentation map array (H, W)
        - T_WC_opencv: Camera pose transformation matrix (4, 4)
        - joint_angles: Robot joint angles array
        - intrinsic_matrix: Camera intrinsic matrix (3, 3)
        
        Optional keys:
        - instance_attribute_maps: Dictionary mapping instance IDs to attributes
        - meta: Additional metadata dictionary
        
        Args:
            sample: Dictionary containing all sample data
        """
        # Start new shard if current one is full
        if self.shard_sample_count >= self.samples_per_shard:
            self._start_new_shard()

        # Generate zero-padded key for this sample
        key = f"{self.sample_counter:06d}"

        # Add required data files
        self._add_rgb_image(f"{key}.rgb.png", sample["rgb"])
        self._add_numpy(f"{key}.depth.npy", sample["depth"])
        self._add_numpy(f"{key}.seg.npy", sample["seg"])

        # Add camera and robot state data
        self._add_numpy(f"{key}.T_WC_opencv.npy", sample["T_WC_opencv"])
        self._add_numpy(f"{key}.joint_angles.npy", sample["joint_angles"])
        self._add_numpy(f"{key}.intrinsic_matrix.npy", sample["intrinsic_matrix"])

        # Add optional metadata files
        if "instance_attribute_maps" in sample:
            self._add_json(f"{key}.instance_attribute_maps.json", sample["instance_attribute_maps"])

        if "meta" in sample:
            self._add_json(f"{key}.json", sample["meta"])

        # Update counters
        self.sample_counter += 1
        self.shard_sample_count += 1

    def close(self):
        """
        Close the current TAR file and clean up resources.
        
        This should be called when finished writing the dataset.
        """
        if self.tar:
            self.tar.close()
            self.tar = None