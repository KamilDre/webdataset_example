import subprocess
import sys
import torch
import tarfile
from PIL import Image
import numpy as np
import json
import io
import random
import matplotlib.pyplot as plt

def read_tar_shard_entries(shard_path, line_limit=21):
    """
    Read the first few entries from a tar shard file.
    
    Args:
        shard_number (int or str): The shard number (e.g., 0 for dataset_shard_000000.tar)
        line_limit (int): Number of lines to output (default: 20)
    
    Returns:
        str: The output from the command, or None if there was an error
    """
    
    # Construct the command
    command = f"cat {shard_path} | dd count=5000 2> /dev/null | tar tf - 2> /dev/null | sed {line_limit}q"
    
    try:
        # Run the command using shell=True since we're using pipes
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30  # Add timeout to prevent hanging
        )
        
        if result.returncode == 0:
            print()
            print(f'Dataset shard {shard_path} content (first {line_limit} lines):\n')
            print(result.stdout)
            return result.stdout
        else:
            print(f"Command failed with return code {result.returncode}", file=sys.stderr)
            if result.stderr:
                print(f"Error: {result.stderr}", file=sys.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print("Command timed out", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error running command: {e}", file=sys.stderr)
        return None


def get_all_tar_shard_sample_ids(shard_path):
    """
    Get all available sample IDs from a tar shard file.
    
    Args:
        shard_path (str): Path to the tar shard file
    
    Returns:
        List[str]: Sorted list of sample IDs (e.g., ['000003', '000004', '000005'])
    """
    sample_ids = set()
    
    try:
        with tarfile.open(shard_path, 'r') as tar:
            for member in tar.getnames():
                if member.count('.') >= 1:  # Ensure it has at least one dot
                    # Extract the numeric prefix (e.g., "000003" from "000003.rgb.png")
                    prefix = member.split('.')[0]
                    sample_ids.add(prefix)
    
    except Exception as e:
        print(f"Error reading tar file {shard_path}: {e}")
        return []
    
    return sorted(list(sample_ids))


def load_sample_from_tar(shard_path: str, sample_id: str):
    """
    Load a specific sample from a tar shard file.
    
    Args:
        shard_path (str): Path to the tar shard file
        sample_id (str): Sample ID to load (e.g., '000003')
    
    Returns:
        Optional[Dict[str, Any]]: Dictionary containing the sample data, or None if not found
    """
    
    try:
        with tarfile.open(shard_path, 'r') as tar:
            sample_data = {}
            
            # Look for all files with this sample_id prefix
            for member in tar:
                if member.isfile() and member.name.startswith(sample_id + '.'):
                    file_data = tar.extractfile(member).read()
                    
                    if member.name.endswith('.rgb.png'):
                        from PIL import Image
                        img = Image.open(io.BytesIO(file_data))
                        sample_data['rgb'] = np.array(img)
                        
                    elif member.name.endswith('.depth.npy'):
                        sample_data['depth'] = np.load(io.BytesIO(file_data))
                        
                    elif member.name.endswith('.seg.npy'):
                        sample_data['seg'] = np.load(io.BytesIO(file_data))
                        
                    elif member.name.endswith('.T_WC_opencv.npy'):
                        sample_data['T_WC_opencv'] = np.load(io.BytesIO(file_data))
                        
                    elif member.name.endswith('.joint_angles.npy'):
                        sample_data['joint_angles'] = np.load(io.BytesIO(file_data))
                        
                    elif member.name.endswith('.intrinsic_matrix.npy'):
                        sample_data['intrinsic_matrix'] = np.load(io.BytesIO(file_data))
                        
                    elif member.name.endswith('.instance_attribute_maps.json'):
                        sample_data['instance_attribute_maps'] = json.loads(file_data.decode('utf-8'))
            
            # Check if we found all 7 expected files
            expected_keys = {'rgb', 'depth', 'seg', 'T_WC_opencv', 'joint_angles', 'intrinsic_matrix', 'instance_attribute_maps'}
            if len(sample_data) == 7 and set(sample_data.keys()) == expected_keys:
                return sample_data
            else:
                print(f"Warning: Sample {sample_id} incomplete. Found {len(sample_data)} files: {list(sample_data.keys())}")
                return sample_data if sample_data else None
                
    except Exception as e:
        print(f"Error loading sample {sample_id} from {shard_path}: {e}")
        return None


def load_random_sample_from_tar(shard_path: str):
    """
    Load a random sample from a tar shard file.
    
    Args:
        shard_path (str): Path to the tar shard file
    
    Returns:
        Optional[Dict[str, Any]]: Dictionary containing the random sample data, or None if error
    """
    
    # Get all available sample IDs
    sample_ids = get_all_tar_shard_sample_ids(shard_path)
    
    if not sample_ids:
        print("No samples found in the tar file")
        return None
    
    # Pick a random sample
    random_sample_id = random.choice(sample_ids)
    print(f"Loading random sample: {random_sample_id}")
    
    return load_sample_from_tar(shard_path, random_sample_id)


def plot_sample_images(sample_data, sample_id="Random Sample"):
    """
    Plot RGB, depth, and segmentation images from a sample.
    
    Args:
        sample_data (dict): Dictionary containing sample data
        sample_id (str): Identifier for the sample (for title)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot RGB image
    if 'rgb' in sample_data:
        axes[0].imshow(sample_data['rgb'])
        axes[0].set_title(f'RGB Image - {sample_id}')
        axes[0].axis('off')
    
    # Plot depth image
    if 'depth' in sample_data:
        depth_img = axes[1].imshow(sample_data['depth'], cmap='viridis')
        axes[1].set_title(f'Depth Map - {sample_id}')
        axes[1].axis('off')
        plt.colorbar(depth_img, ax=axes[1], shrink=0.6)
    
    # Plot segmentation image
    if 'seg' in sample_data:
        seg_img = axes[2].imshow(sample_data['seg'], cmap='tab20')
        axes[2].set_title(f'Segmentation - {sample_id}')
        axes[2].axis('off')
        plt.colorbar(seg_img, ax=axes[2], shrink=0.6)
    
    plt.tight_layout()
    plt.show()


def print_and_pass(sample_dict):
    print("Sample dictionary:")
    for key, value in sample_dict.items():
        print(f"  Key '{key}': Type={type(value)}")
    return sample_dict

def inspect_dataloader_output(data_loader, num_batches_to_inspect=2):


    print(f"\n--- Starting data inspection ---")

    
    for batch_idx, sample_batch_dict in enumerate(data_loader):
        if batch_idx >= num_batches_to_inspect:
            break
        
        print(f"\n--- Inspecting Batch {batch_idx + 1} ---")
        print_and_pass(sample_batch_dict)  # Print keys and types of the sample batch dict

        # sample_batch_dict is now a dictionary, where each key maps to a batched component
        print(f"Batch contains {len(sample_batch_dict)} components.\n")

        # --- RGB ---
        if 'rgb' in sample_batch_dict:
            rgb_images = sample_batch_dict['rgb']
            print(f"  RGB: Type={type(rgb_images)}, Batch size={len(rgb_images)}")
            if len(rgb_images) > 0 and isinstance(rgb_images[0], Image.Image):
                print(f"       Image 0 size: {rgb_images[0].size}, Mode: {rgb_images[0].mode}")
                if len(rgb_images) > 1:
                    print(f"       Image 1 size: {rgb_images[1].size}, Mode: {rgb_images[1].mode}")
            elif len(rgb_images) > 0 and isinstance(rgb_images, torch.Tensor):
                print(f"       RGB batch shape: {rgb_images.shape}, Dtype: {rgb_images.dtype}")
                print(f"       RGB batch min: {rgb_images.min():.4f}, max: {rgb_images.max():.4f}")
            else:
                print("  RGB: Unexpected type or empty batch.")
            print()
            
        # Uncomment and modify as needed when you add more data types
        # --- DEPTH ---
        if 'depth' in sample_batch_dict:
            depth_maps = sample_batch_dict['depth']
            print(f"  Depth: Type={type(depth_maps)}, Batch size={depth_maps.shape[0] if isinstance(depth_maps, np.ndarray) else len(depth_maps)}")
            if isinstance(depth_maps, np.ndarray):
                print(f"         Depth batch shape: {depth_maps.shape}, Dtype: {depth_maps.dtype}")
                print(f"         Depth batch min: {depth_maps.min():.4f}, max: {depth_maps.max():.4f}")
            elif isinstance(depth_maps, torch.Tensor):
                print(f"         Depth batch shape: {depth_maps.shape}, Dtype: {depth_maps.dtype}")
                print(f"         Depth batch min: {depth_maps.min():.4f}, max: {depth_maps.max():.4f}")
            else:
                print("  Depth: Unexpected type or empty batch.")
            print()

        # --- SEGMENTATION ---
        if 'seg' in sample_batch_dict:
            seg_maps = sample_batch_dict['seg']
            print(f"  Segmentation: Type={type(seg_maps)}, Batch size={len(seg_maps)}")
            if len(seg_maps) > 0 and isinstance(seg_maps[0], Image.Image):
                print(f"              Seg 0 size: {seg_maps[0].size}, Mode: {seg_maps[0].mode}")
            elif len(seg_maps) > 0 and isinstance(seg_maps, torch.Tensor):
                print(f"              Seg batch shape: {seg_maps.shape}, Dtype: {seg_maps.dtype}")
                print(f"              Unique classes: {seg_maps[0].unique()}")
            else:
                print("  Segmentation: Unexpected type or empty batch.")
            print()

        # --- INTRINSIC MATRIX ---
        if 'intrinsic_matrix' in sample_batch_dict:
            intrinsics = sample_batch_dict['intrinsic_matrix']
            print(f"  Intrinsics: Type={type(intrinsics)}, Batch size={intrinsics.shape[0] if isinstance(intrinsics, np.ndarray) else len(intrinsics)}")
            if isinstance(intrinsics, np.ndarray) or isinstance(intrinsics, torch.Tensor):
                print(f"              Intrinsic batch shape: {intrinsics.shape}")
                print(f"              Intrinsic batch values (first):\n{intrinsics[0]}")
            
            else:
                print("  Intrinsics: Unexpected type or empty batch.")
            print()

        # --- EXTRINSIC MATRIX ---
        if 'T_WC_opencv' in sample_batch_dict:
            extrinsics = sample_batch_dict['T_WC_opencv']
            print(f"  Extrinsics: Type={type(extrinsics)}, Batch size={extrinsics.shape[0] if isinstance(extrinsics, np.ndarray) else len(extrinsics)}")
            if isinstance(extrinsics, np.ndarray) or isinstance(extrinsics, torch.Tensor):
                print(f"              Extrinsic batch shape: {extrinsics.shape}")
                print(f"              Extrinsic batch values (first):\n{extrinsics[0]}")
            else:
                print("  Extrinsics: Unexpected type or empty batch.")
            print()

        # --- JOINT ANGLES ---
        if 'joint_angles' in sample_batch_dict:
            joint_angles = sample_batch_dict['joint_angles']
            print(f"  Joint Angles: Type={type(joint_angles)}, Batch size={joint_angles.shape[0] if isinstance(joint_angles, np.ndarray) else len(joint_angles)}")
            if isinstance(joint_angles, np.ndarray) or isinstance(joint_angles, torch.Tensor):
                print(f"                Joint Angles batch shape: {joint_angles.shape}")
                print(f"                Joint Angles batch values (first): {joint_angles[0]}")
            else:
                print("  Joint Angles: Unexpected type or empty batch.")
            print()

        # # --- INSTANCE ATTRIBUTE MAPS ---
        # if 'instance_attribute_maps' in sample_batch_dict:
        #     attribute_maps = sample_batch_dict['instance_attribute_maps']
        #     print(f"  Instance Attribute Maps: Type={type(attribute_maps)}, Batch size={len(attribute_maps)}")
        #     if len(attribute_maps) > 0 and isinstance(attribute_maps[0], list):
        #         print(f"                           Map 0: {attribute_maps[0]}")
        #     else:
        #         print(f"                           Unexpected type or empty batch.")
        #     print()

    print("\n--- Data inspection complete ---")