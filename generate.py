import numpy as np
from writer import WebDatasetWriter

if __name__ == '__main__':

    # Initialize WebDatasetWriter
    data_writer = WebDatasetWriter(output_dir="../data", samples_per_shard=100)

    for _ in range(1000):
        # Collect data for the current frame
        sample_data = {
            "rgb": np.random.rand(256, 256, 3),
            "depth": np.random.uniform(low=0, high=5, size=(256, 256, 1)),
            "seg": np.uint8(np.random.uniform(size=(256, 256)) > 0.5),
            "intrinsic_matrix": np.random.rand(3, 3), # This is constant per batch in this revised code
            "T_WC_opencv": np.random.rand(4, 4),
            "joint_angles": np.random.randn(2, 2),
            "instance_attribute_maps": [{'idx': 0, 'name': 'Background'}, {'idx': 1, 'name': 'Foreground'}]
        }
        data_writer.add_sample(sample_data)

    # Close the writer
    data_writer.close()