import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image
import numpy as np
from tqdm import tqdm
import h5py
from diffsynth.trainers.skeleton_tfs import *
import glob
import plotly.graph_objects as go

def vis_3d(root, indices):
    # root = h5py.File(input_file)
    
    # Prepare data for animation
    frames = []
    QUERY_TFS = LEFT_ARM + LEFT_FINGERS + RIGHT_ARM + RIGHT_FINGERS + NECK
    # extract position data 
    
    for i in range(len(indices)):
        tfs = np.zeros([len(QUERY_TFS), 3]) 
        for j, tf_name in enumerate(QUERY_TFS):
            tfs[j] = root['/transforms/' + tf_name][indices[i]][:3, 3]
        frames.append(tfs.copy())
    
    return frames

def to_tensor_safe(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(device=device, dtype=dtype)
    else:  # assume numpy array or list
        return torch.as_tensor(x, device=device, dtype=dtype).detach().clone()

def find_and_load_metadata(root_path):
    """
    Finds all 'metadata_vace.csv' files in a directory tree, reads them,
    and returns a single list of data entries with absolute paths.
    """
    csv_files = glob.glob(os.path.join(root_path, '**', 'metadata_vace.csv'), recursive=True)
    print(f"Found {len(csv_files)} metadata CSV files.")

    all_data_rows = []
    counter = 0
    for csv_file in tqdm(csv_files, desc="Loading metadata..."):
        subfolder_path = os.path.dirname(csv_file)
        df = pandas.read_csv(csv_file)

        for _, row in df.iterrows():
            data_dict = row.to_dict()
            counter += 1
            for key in ['video', 'prompt','vace_video', 'pose', 'vace_reference_image']:
                if key in data_dict and isinstance(data_dict[key], str):
                    data_dict[key] = os.path.join(subfolder_path, data_dict[key])
            all_data_rows.append(data_dict)
    print(f'counter={counter}')
    return all_data_rows

def index_episodes(dataset_path): 
    # find all hdf5 files
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_path):
        for filename in fnmatch.filter(files, "*.hdf5"):
            hdf5_files.append(os.path.join(root, filename))
    print(f"Found {len(hdf5_files)} hdf5 files")

    # get lengths of all hdf5 files
    all_episode_len = []
    for dataset_path in tqdm(hdf5_files, desc='iterating dataset_path to get all episode lengths...'):
        try:
            with h5py.File(dataset_path, "r") as root:
                action = root['/transforms/leftHand'][()]
        except Exception as e:
            print(f"Error loading {dataset_path}")
        all_episode_len.append(len(action))
    
    return hdf5_files, all_episode_len

def rotation_matrix_to_euler(R):
    """Convert rotation matrices [B,3,3] → Euler angles [B,3] (yaw, pitch, roll)"""
    sy = torch.sqrt(R[:, 0, 0]**2 + R[:, 1, 0]**2)
    singular = sy < 1e-6

    yaw = torch.where(~singular, torch.atan2(R[:, 1, 0], R[:, 0, 0]), torch.zeros_like(sy))   # Yaw (around Z)
    pitch = torch.where(~singular, torch.atan2(-R[:, 2, 0], sy), torch.zeros_like(sy))        # Pitch (around X)
    roll = torch.where(~singular, torch.atan2(R[:, 2, 1], R[:, 2, 2]), torch.zeros_like(sy))  # Roll (around Y)
    return torch.stack((yaw, pitch, roll), dim=1)  # [B,3]: yaw(Z), pitch(X), roll(Y)

MOTION_LABELS = ["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown", "In", "Out", "Static"]

label_to_id = {name: i for i, name in enumerate(MOTION_LABELS)}

def classify_camera_motion(cam_ext, thresh_trans=0.001, thresh_rot=0.001):
    """
    Classify camera motion between consecutive frames and compute motion magnitudes.

    Args:
        cam_ext: [L, 4, 4] torch.Tensor - camera extrinsic matrices
        thresh_trans: float - translation threshold
        thresh_rot: float - rotation threshold (radians)

    Returns:
        labels:      [L-1] torch.LongTensor  - motion class IDs
        magnitudes:  [L-1] torch.FloatTensor - per-step motion magnitudes
    """
    L = cam_ext.shape[0]

    # Relative transformation between consecutive frames
    rel = cam_ext[1:] @ torch.linalg.inv(cam_ext[:-1])  # [L-1,4,4]
    t = rel[:, :3, 3]                                  # [L-1,3]
    R = rel[:, :3, :3]                                 # [L-1,3,3]

    # Convert rotation matrix difference to Euler angles
    euler = rotation_matrix_to_euler(R)                # [L-1,3] (yaw, pitch, roll)

    # Translation and rotation magnitudes between consecutive frames
    trans_mag = torch.linalg.norm(t, dim=1)            # per-step translation magnitude
    rot_mag = torch.linalg.norm(euler, dim=1)          # per-step rotation magnitude
    magnitudes = torch.sqrt(trans_mag**2 + rot_mag**2) # combine into one scalar per frame

    labels = []
    for i in range(L - 1):
        dx, dy, dz = t[i]
        yaw, pitch, roll = euler[i]

        horiz = ""
        vert = ""
        depth = ""

        # Horizontal motion (translation X or yaw rotation)
        if dx < -thresh_trans or yaw > thresh_rot:
            horiz = "Left"
        elif dx > thresh_trans or yaw < -thresh_rot:
            horiz = "Right"

        # Vertical motion (translation Y or pitch rotation)
        if dy > thresh_trans or pitch > thresh_rot:
            vert = "Up"
        elif dy < -thresh_trans or pitch < -thresh_rot:
            vert = "Down"

        # Depth motion (translation Z)
        if dz < -thresh_trans:
            depth = "In"
        elif dz > thresh_trans:
            depth = "Out"

        # Combine label priority
        if horiz and vert:
            label = horiz + vert
        elif horiz:
            label = horiz
        elif vert:
            label = vert
        elif depth:
            label = depth
        else:
            label = "Static"

        labels.append(label_to_id[label])

    return torch.tensor(labels, dtype=torch.long), magnitudes



class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image



class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image



class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    


class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def get_video_frame_indices(self, total_frames, target_frames):
        if total_frames >= target_frames:
            # Uniform sampling between first and last frame
            frame_indices = np.linspace(
                0,
                total_frames - 1,
                target_frames
            )
            frame_indices = np.round(frame_indices).astype(int)

        else:
            # Not enough frames: uniformly sample, then tile
            base_indices = np.linspace(
                0,
                total_frames - 1,
                total_frames
            )
            base_indices = np.round(base_indices).astype(int)

            repeat_factor = int(np.ceil(target_frames / total_frames))
            tiled = np.tile(base_indices, repeat_factor)

            frame_indices = tiled[:target_frames]

        return frame_indices
        
    def __call__(self, data: str):
        ### original 
        # reader = imageio.get_reader(data)
        # num_frames = self.get_num_frames(reader)
        # frames = []
        # for frame_id in range(num_frames):
        #     frame = reader.get_data(frame_id)
        #     frame = Image.fromarray(frame)
        #     frame = self.frame_processor(frame)
        #     frames.append(frame)

        ## backward filling
        # reader = imageio.get_reader(data)
        # total_frames = int(reader.count_frames())
        # num_frames_to_load = self.get_num_frames(reader)
        # frames = []
        # # --- Case 1: Video has enough frames ---
        # if total_frames >= self.num_frames:
        #     for frame_id in range(num_frames_to_load):
        #         frame = reader.get_data(frame_id)
        #         frame = Image.fromarray(frame)
        #         frame = self.frame_processor(frame)
        #         frames.append(frame)

        # # --- Case 2: Video shorter than target num_frames ---
        # else:
        #     # Load all available frames first
        #     for frame_id in range(num_frames_to_load):
        #         frame = reader.get_data(frame_id)
        #         frame = Image.fromarray(frame)
        #         frame = self.frame_processor(frame)
        #         frames.append(frame)

        #     # Compute how many more frames we need to fill
        #     missing = self.num_frames - len(frames)

        #     # Read backward from the end to the start to fill missing frames
        #     for i in range(missing):
        #         frame_id = total_frames - 1 - (i % total_frames)
        #         frame = reader.get_data(frame_id)
        #         frame = Image.fromarray(frame)
        #         frame = self.frame_processor(frame)
        #         frames.append(frame)


        # uniform sampling
        reader = imageio.get_reader(data)
        total_frames = int(reader.count_frames())

        frame_indices = self.get_video_frame_indices(total_frames, self.num_frames)

        frames = []
        for idx in frame_indices:
            frame = reader.get_data(int(idx))
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)

        
    
        reader.close()
        return frames



class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]



class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)



class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        # self.load_metadata(metadata_path)
        self.data = find_and_load_metadata(base_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(num_frames, time_division_factor, time_division_remainder) >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def get_camera_extrinsic_indices(self, num_available_frames, target_frames):
   
        if num_available_frames >= target_frames:
            cam_indices = np.linspace(
                0,
                num_available_frames - 1,
                target_frames
            )
            indices = np.round(cam_indices).astype(int)

        else:
            base_indices = np.linspace(
                0,
                num_available_frames - 1,
                num_available_frames
            )
            base_indices = np.round(base_indices).astype(int)

            repeat_factor = int(np.ceil(target_frames / num_available_frames))
            tiled_indices = np.tile(base_indices, repeat_factor)

            indices = tiled_indices[:target_frames]

        return indices

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            original_video_path = data.get('video')
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key]
                    elif key in self.data_file_keys:
                        if key == 'video':
                           data[key] = self.main_data_operator(original_video_path)
                           self.num_frame = len(data[key])
                        elif key == 'vace_reference_image':
                           data[key] = self.main_data_operator(data.get(key))[0]
                           data['reference_image'] = data[key]
                        elif key == 'vace_video':
                            try:
                               data[key] = self.main_data_operator(data.get(key))
                            except Exception as e:
                               print(f"[WARN] Failed to load vace_video: {data.get(key)}")
                               data[key] = self.main_data_operator(original_video_path)
                        elif key == 'prompt':
                           data[key] = data.get(key)
                        elif key == 'pose':
                           hdf5_file = data.get(key)
                           query_tfs = WRISTS
                           with h5py.File(hdf5_file, "r") as root:
                               available_cam_ext = root['/transforms/camera'][:]
                               num_available_frames = available_cam_ext.shape[0]

                               if num_available_frames == 0:
                                  raise ValueError(f"HDF5 file {hdf5_file} contains no frames.")

                               indices = np.arange(self.num_frame) % num_available_frames

                               # --- match video frame indices ---
                               # num_frames_to_load must be SAME as video sampling
                               # --- If the video has enough frames ---
                               indices = self.get_camera_extrinsic_indices(
                                    num_available_frames,
                                    self.num_frame
                               )

                               tfdtype = available_cam_ext.dtype
                               tfs = np.zeros([len(query_tfs), self.num_frame, 4, 4], dtype=tfdtype)
                               for i, tf_name in enumerate(query_tfs):
                                   available_tfs = root['/transforms/' + tf_name][:]
                                   tfs[i] = available_tfs[indices]

                               keypoints_for_3d_vis = vis_3d(root, indices)
                               cam_ext = available_cam_ext[indices]
                               cam_int = root['/camera/intrinsic'][:] # Intrinsics are usually static
                               if root.attrs['llm_type'] == 'reversible':
                                  direction = root.attrs['which_llm_description']
                                  lang_instruct = root.attrs['llm_description' if direction == '1' else 'llm_description2'] 
                               else:
                                  lang_instruct = root.attrs['llm_description'] 
                               
                               if 'confidences' in root:
                                   confs = np.zeros([len(query_tfs), self.num_frame], dtype=tfdtype)
                                   for i, tf_name in enumerate(query_tfs):
                                       available_confs = root['/confidences/' + tf_name][:]
                                       confs[i] = available_confs[indices]
                               else:
                                   confs = np.zeros([len(query_tfs), self.num_frame], dtype=tfdtype)
                                
                               cam_motion_input = cam_ext[-1:]
                               cam_motion_input_new = torch.cat([to_tensor_safe(cam_ext), to_tensor_safe(cam_motion_input)], dim=0)
                               cam_motion, magnitude = classify_camera_motion(cam_motion_input_new)
                               data['camera_control_direction'] = cam_motion
                               data['camera_control_speed'] = magnitude
                    
                               data[key] = {'tfs': tfs, 'cam_ext': cam_ext, 'cam_int': cam_int, 'lang_instruct': lang_instruct, 'confs': confs, 'path': hdf5_file,'3d_vis': keypoints_for_3d_vis}

        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
