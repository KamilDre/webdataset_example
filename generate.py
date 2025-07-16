import blenderproc as bproc
import argparse
import bpy
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import logging

sys.path.append('/')
from sim2real_datagen.writer import WebDatasetWriter
from sim2real_datagen.utils import pose_inv, calculate_intrinsic_for_new_resolution, add_world_coordinate_system_opencv
from sim2real_datagen.globals import RENDERING_INTRINSIC, RENDERING_IMAGE_WIDTH, RENDERING_IMAGE_HEIGHT, T_WC_opengl, T_WC_opencv, SLEW_RANGE, LUFF_RANGE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Define base camera properties to be randomized
BASE_INTRINSIC = RENDERING_INTRINSIC
BASE_T_WC_OPENGL = T_WC_opengl

# Define the maximum number of lights
MAX_LIGHTS = 4 

def randomize_robot_joints(robot, frame):
    """
    Randomize robot joints in allowed ranges for links[1] and links[2].
    """
    slew_angle = np.random.uniform(low=SLEW_RANGE[0], high=SLEW_RANGE[1])
    luff_angle = np.random.uniform(low=LUFF_RANGE[0], high=LUFF_RANGE[1])

    logger.debug(f"Randomizing joints: slew={slew_angle:.3f}, luff={luff_angle:.3f}")

    robot.set_rotation_euler_fk(link=robot.links[1], rotation_euler=slew_angle, mode='absolute', frame=frame)
    robot.set_rotation_euler_fk(link=robot.links[2], rotation_euler=luff_angle, mode='absolute', frame=frame)

    return slew_angle, luff_angle


def randomize_camera_extrinsics(base_extrinsics, translation_std=0.03, rotation_deg_std=5):
    """
    Randomize camera extrinsic matrix by small translations and rotations around base_extrinsics.
    Args:
        base_extrinsics (np.ndarray): 4x4 base camera pose matrix (world->camera or camera->world).
        translation_std (float): standard deviation in meters for translation noise.
        rotation_deg_std (float): std deviation degrees for rotation noise.
    Returns:
        np.ndarray: randomized 4x4 extrinsic matrix.
    """
    # Random translation noise
    delta_t = np.random.normal(scale=translation_std, size=3)
    # Random rotation noise
    delta_r = R.from_euler('xyz', np.random.normal(scale=rotation_deg_std, size=3), degrees=True).as_matrix()

    randomized = np.eye(4)
    randomized[:3, :3] = base_extrinsics[:3, :3] @ delta_r
    randomized[:3, 3] = base_extrinsics[:3, :3] @ delta_t + base_extrinsics[:3, 3]

    logger.debug(f"Randomized extrinsics translation delta: {delta_t}")
    logger.debug(f"Randomized extrinsics rotation delta (deg): {rotation_deg_std}")

    return randomized


def randomize_camera_intrinsics(base_intrinsics, focal_std=5.0, principal_point_std=2.0):
    """
    Randomize intrinsic camera matrix elements slightly.
    Args:
        base_intrinsics (np.ndarray): 3x3 base intrinsic matrix.
        focal_std (float): std deviation in pixels for focal length noise.
        principal_point_std (float): std deviation in pixels for principal point noise.
    Returns:
        np.ndarray: randomized 3x3 intrinsic matrix.
    """
    K = base_intrinsics.copy()

    fx_noise = np.random.normal(scale=focal_std)
    fy_noise = np.random.normal(scale=focal_std)
    cx_noise = np.random.normal(scale=principal_point_std)
    cy_noise = np.random.normal(scale=principal_point_std)

    K[0, 0] += fx_noise
    K[1, 1] += fy_noise
    K[0, 2] += cx_noise
    K[1, 2] += cy_noise

    logger.debug(f"Randomized intrinsics fx noise: {fx_noise:.2f}, fy noise: {fy_noise:.2f}")
    logger.debug(f"Randomized intrinsics cx noise: {cx_noise:.2f}, cy noise: {cy_noise:.2f}")

    return K

def initialize_lights(max_lights=MAX_LIGHTS):
    """
    Initializes and returns a list of light objects up to max_lights.
    These lights will be reused across frames/batches.
    """
    lights = []
    for i in range(max_lights):
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_name(f"DynamicLight_{i}") # Give unique names for easier debugging
        lights.append(light)
    logger.info(f"Initialized {max_lights} persistent light objects.")
    return lights


def randomize_and_set_lights(frame_idx, lights, min_lights=1, max_lights=MAX_LIGHTS, position_range=5, energy_range=(1000, 4000)):
    """
    Randomizes properties of the provided persistent lights.
    Sets energy to 0 for lights exceeding the random number of active lights.
    Args:
        frame_idx (int): The current frame index.
        lights (List[bproc.types.Light]): The list of pre-initialized light objects.
        min_lights (int): minimum number of active lights.
        max_lights (int): maximum number of active lights.
        position_range (float): range in meters for random positions in each axis (+/-).
        energy_range (tuple): (min_energy, max_energy) range for light energy.
    """
    num_active_lights = np.random.randint(min_lights, max_lights + 1)
    
    logger.debug(f"Randomizing {num_active_lights} active lights for frame {frame_idx} (out of {len(lights)} total)")

    for i, light in enumerate(lights):
        if i < num_active_lights:
            # Activate and randomize this light
            pos = np.random.uniform(low=-position_range, high=position_range, size=3)
            energy = np.random.uniform(*energy_range)
            light.set_location(pos, frame_idx)
            light.set_energy(energy, frame_idx)
            logger.debug(f"Light {i} pos: {pos}, energy: {energy:.1f}")
        else:
            # Deactivate this light by setting energy to 0
            light.set_energy(0.0, frame_idx) # Set energy to 0 to effectively turn it off
            logger.debug(f"Light {i} deactivated (energy set to 0)")


def randomize_materials(robot):
    """
    Randomize materials for all visuals in robot links.
    Example: vary roughness, metallic, specular IOR.
    """
    for link in robot.links:
        if not link.visuals:
            continue
        for mat in link.visuals[0].get_materials():
            roughness = np.random.uniform(0.1, 0.5)
            metallic = np.random.uniform(0.0, 1.0)
            specular_ior = np.random.uniform(0.5, 1.0)

            mat.set_principled_shader_value("Roughness", roughness)
            mat.set_principled_shader_value("Metallic", metallic)
            mat.set_principled_shader_value("Specular IOR Level", specular_ior)

            logger.debug(f"Material updated for {link.get_name()}: Roughness={roughness:.2f}, Metallic={metallic:.2f}, Specular IOR={specular_ior:.2f}")


def convert_between_opengl_opencv(T):
    return T @ np.array([[1., 0., 0., 0.],
                         [0., - 1., 0., 0.],
                         [0., 0., - 1., 0.],
                         [0., 0., 0., 1.]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', nargs='?', default="/data", help="Path to where the final files will be saved")
    parser.add_argument('--num_frames', type=int, default=6, help="Total number of frames to render")
    parser.add_argument('--batch_size', type=int, default=3, help="Number of frames to render at once")
    parser.add_argument('--samples_per_shard', type=int, default=3, help="Number of frames to render at once")
    parser.add_argument('--urdf_file', nargs='?', default="/final_model/trustline_crane.urdf", help="Path to the .urdf file")
    args = parser.parse_args()

    # Initialize WebDatasetWriter
    data_writer = WebDatasetWriter(output_dir=args.dataset_dir, samples_per_shard=args.samples_per_shard)

    bproc.init()

    robot = bproc.loader.load_urdf(urdf_file=args.urdf_file)
    robot.set_ascending_category_ids()

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by=["instance", "name"])

    num_iter = int(args.num_frames / args.batch_size)
    
    # Initialize the maximum number of lights once at the beginning
    persistent_lights = initialize_lights(max_lights=MAX_LIGHTS)

    for _ in range(num_iter):
        # Reset keyframe index for the next iteration (BlenderProc automatically increments it)
        bproc.utility.reset_keyframes()

        # Randomize camera intrinsics (only once per batch if they are constant for the batch)
        randomized_intrinsics = randomize_camera_intrinsics(BASE_INTRINSIC)
        bproc.camera.set_intrinsics_from_K_matrix(randomized_intrinsics, RENDERING_IMAGE_WIDTH, RENDERING_IMAGE_HEIGHT)

        slew_luff_angles = []
        camera_poses_opencv = []

        for frame_idx in range(args.batch_size):
            # Randomize robot joint angles
            slew, luff = randomize_robot_joints(robot, frame_idx)

            # Randomize camera extrinsics
            T_WC_opengl_random = randomize_camera_extrinsics(BASE_T_WC_OPENGL)
            bproc.camera.add_camera_pose(cam2world_matrix=T_WC_opengl_random, frame=frame_idx)
            
            # Randomize and set properties for the persistent lights for the current frame
            randomize_and_set_lights(frame_idx, persistent_lights)

            # # Randomize materials (if you want them to change per frame)
            # randomize_materials(robot)

            slew_luff_angles.append([slew, luff])
            camera_poses_opencv.append(convert_between_opengl_opencv(T_WC_opengl_random))

        # Render data for the current batch of frames
        data = bproc.renderer.render()

        for frame_idx in range(args.batch_size):

            data["depth"][frame_idx][data["depth"][frame_idx] == 1e10] = 0  # Set background depth values to 0

            # Collect data for the current frame
            sample_data = {
                "rgb": data["colors"][frame_idx],
                "depth": data["depth"][frame_idx],
                "seg": data["instance_segmaps"][frame_idx].astype(np.uint8),
                "intrinsic_matrix": randomized_intrinsics, # This is constant per batch in this revised code
                "T_WC_opencv": camera_poses_opencv[frame_idx],
                "joint_angles": np.array(slew_luff_angles[frame_idx]),
                "instance_attribute_maps": data['instance_attribute_maps'][frame_idx] 
            }
            data_writer.add_sample(sample_data)


    # Close the writer
    data_writer.close()