import os
import numpy as np
from loader import get_loader
from webdataset_utils import read_tar_shard_entries, load_random_sample_from_tar, inspect_dataloader_output, plot_sample_images


def main():
    # Dataset configuration
    dataset_dir = "/data"
    num_shards = 2  # Adjust based on your actual generated shard count
    
    # Create tar pattern for webdataset to read multiple shards
    # This will match dataset_shard_000000.tar, dataset_shard_000001.tar, etc.
    tar_pattern = os.path.join(dataset_dir, f"dataset_shard_{{0..{num_shards-1:06d}}}.tar")
    
    # DataLoader configuration
    example_shard_number = 1
    batch_size = 5
    shuffle_buffer = 10
    num_workers = 1

    # Create the webdataset DataLoader
    data_loader = get_loader(tar_pattern,
                             batch_size=batch_size, 
                             shuffle_buffer=shuffle_buffer,
                             num_workers=num_workers)
    
    # ============================================
    # PART 1: INSPECT THE TAR FILE DIRECTLY
    # ============================================
    
    # Construct path to specific shard for direct inspection
    shard_path = f'/data/dataset_shard_{example_shard_number:06d}.tar'
    
    # Read and display the first few entries in the tar file
    # This shows the raw file structure (e.g., 000003.rgb.png, 000003.depth.npy, etc.)
    read_tar_shard_entries(shard_path)

    # Load a random sample directly from the tar file
    # This bypasses the DataLoader and reads the raw data
    random_sample = load_random_sample_from_tar(shard_path)
    if random_sample:
        print("=" * 50)
        print("DIRECT TAR FILE INSPECTION - Random Sample:")
        print("=" * 50)
        print(f"  RGB shape: {random_sample['rgb'].shape}")
        print(f"  Depth shape: {random_sample['depth'].shape}")
        print(f"  Segmentation shape: {random_sample['seg'].shape}")
        print(f"  Transform shape: {random_sample['T_WC_opencv'].shape}")
        print(f"  Joint angles shape: {random_sample['joint_angles'].shape}")
        print(f"  Intrinsic matrix shape: {random_sample['intrinsic_matrix'].shape}")
        print(f"  Instance attributes: {random_sample['instance_attribute_maps']}")
        
        # Plot the RGB, depth, and segmentation images
        print("\nPlotting sample images...")
        plot_sample_images(random_sample)
    
    # ============================================
    # PART 2: INSPECT THE DATALOADER OUTPUT
    # ============================================
    
    # Inspect how the DataLoader processes and batches the data
    # This shows the transformed/batched data as it would be fed to a model
    print("\n" + "=" * 50)
    print("DATALOADER INSPECTION:")
    print("=" * 50)
    inspect_dataloader_output(data_loader)
    
if __name__ == '__main__':
    main()