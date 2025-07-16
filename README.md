# WebDataset Example

This folder contains a complete example of using WebDataset for machine learning datasets, specifically designed for robotics/computer vision applications. WebDataset is a high-performance format that stores samples in TAR archives, making it ideal for large-scale datasets with multiple data modalities.

## Files Overview

### Core Components

**`writer.py`** - WebDataset Writer Class
- Contains the `WebDatasetWriter` class for creating WebDataset TAR shards
- Handles automatic sharding (splitting data across multiple TAR files)
- Supports multiple data types: RGB images, depth maps, segmentation, camera poses, joint angles, and metadata
- Includes resumable writing (can continue from existing shards)
- Optimized for robotics datasets with image + pose + metadata combinations

**`loader.py`** - PyTorch DataLoader
- Provides `get_loader()` function that creates a PyTorch DataLoader from WebDataset TAR files
- Handles automatic decoding of images and numpy arrays
- Includes shuffling, batching, and multi-worker support
- Converts data to PyTorch tensors where appropriate
- Supports glob patterns for loading multiple shards

**`webdataset_utils.py`** - Utility Functions
- `read_tar_shard_entries()` - Inspect TAR file contents without loading
- `load_sample_from_tar()` - Load specific samples by ID
- `load_random_sample_from_tar()` - Load random samples for testing
- `plot_sample_images()` - Visualize RGB, depth, and segmentation data
- `inspect_dataloader_output()` - Debug DataLoader batching behavior

### Testing and Examples

**`test_dataset.py`** - Complete Testing Example
- Demonstrates both direct TAR file inspection and DataLoader usage
- Shows how to configure paths and parameters
- Includes visualization of loaded samples
- Comprehensive testing of the entire pipeline

**`generate.py`** - Data Generation Example (Blender/BlenderProc)
- Example of using the WebDatasetWriter in a real application
- Shows how to generate synthetic robotics data using BlenderProc
- Demonstrates proper sample formatting and batch processing
- Includes camera randomization, lighting, and material variations

### How to use

```
from loader import get_loader

# Load dataset
data_loader = get_loader("dataset_shard_{000000..000010}.tar", batch_size=8)

# Training loop
for batch in data_loader:
    rgb = batch['rgb']          # [B, C, H, W] tensor
    depth = batch['depth']      # [B, 1, H, W] tensor
    poses = batch['T_WC_opencv'] # [B, 4, 4] array
    # ... train your model
```

## Data Format (my application)

Each sample in the dataset contains:
- **RGB image** (`.rgb.png`) - Color image as PNG
- **Depth map** (`.depth.npy`) - Depth values as numpy array
- **Segmentation** (`.seg.npy`) - Segmentation mask as numpy array
- **Camera pose** (`.T_WC_opencv.npy`) - 4x4 transformation matrix
- **Joint angles** (`.joint_angles.npy`) - Robot joint configurations
- **Intrinsics** (`.intrinsic_matrix.npy`) - Camera intrinsic matrix
- **Metadata** (`.instance_attribute_maps.json`) - Instance attributes

## Customization

To adapt this for your dataset:

1. **Modify `writer.py`**: Update `add_sample()` to match your data structure
2. **Update `loader.py`**: Adjust the `convert_sample()` function for your data types
3. **Change file extensions**: Update the decode patterns in the loader
4. **Add new data types**: Extend the writer with additional `_add_*()` methods

## Learn More

For detailed information about WebDataset:

- **Main GitHub Repository**: https://github.com/webdataset/webdataset
- **Video Tutorials**:
  - [WebDataset Overview](https://www.youtube.com/watch?v=kNuA2wflygM)
  - [Advanced Usage](https://www.youtube.com/watch?v=mTv_ePYeBhs)
  - [Performance Tips](https://www.youtube.com/watch?v=v_PacO-3OGQ)
  - [Integration Examples](https://www.youtube.com/watch?v=kIv8zDpRUec)

## Notes

- The `generate.py` file shows the writer usage but requires BlenderProc and the crane model
- You may need to adjust import paths depending on your project structure etc.
