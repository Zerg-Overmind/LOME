import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image
from tqdm import tqdm
import h5py
import numpy as np
from diffsynth.trainers.skeleton_tfs import WRISTS
import glob

def find_and_load_metadata(root_path):
    """
    Finds all 'metadata_vace.csv' files in a directory tree, reads them,
    and returns a single list of data entries with absolute paths.
    """
    csv_files = glob.glob(os.path.join(root_path, '**', 'metadata_vace.csv'), recursive=True)
    print(f"Found {len(csv_files)} metadata CSV files.")

    all_data_rows = []
    for csv_file in tqdm(csv_files, desc="Loading metadata..."):
        subfolder_path = os.path.dirname(csv_file)
        df = pandas.read_csv(csv_file)

        for _, row in df.iterrows():
            data_dict = row.to_dict()
            for key in ['video', 'prompt','vace_video', 'pose', 'vace_reference_image']:
                if key in data_dict and isinstance(data_dict[key], str):
                    data_dict[key] = os.path.join(subfolder_path, data_dict[key])
            all_data_rows.append(data_dict)

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
        #total_frames = len(reader)
        #if total_frames < num_frames:
        if int(reader.count_frames()) < num_frames:
            #num_frames = total_frames
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frame
    def __call__(self, data: str):
        frames = []
        reader = None
        try:
            reader = imageio.get_reader(data)
            total_frames = reader.count_frames()

            if total_frames >= self.num_frames:
                indices = range(self.num_frames)
            else:
                indices = np.arange(self.num_frames) % total_frames

            for idx in indices:
                frame = reader.get_data(idx)
                frames.append(self.frame_processor(Image.fromarray(frame)))

        finally:
            if reader:
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

class ToTensor(DataProcessingOperator):
    """Converts a PIL Image or numpy.ndarray to a tensor."""
    def __init__(self):
        self.op = torchvision.transforms.ToTensor()

    def __call__(self, image):
        return self.op(image)

class SelectFirstItem(DataProcessingOperator):
    """Takes a list and returns only the first item."""
    def __call__(self, data: list):
        if not data:
            return None
        return data[0]

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
        #self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None

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
        final_processor = SequencialProcess(ToTensor())
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList() >> final_processor),
                (("gif",), LoadGIF(num_frames, time_division_factor, time_division_remainder) >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> final_processor),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                ) >> final_processor),
            ])),
        ])


    @staticmethod
    def default_first_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        """
        Processes a video file path but outputs a single, processed image 
        (the first frame), mirroring the structure of default_image_operator.
        """
        # Define the core pipeline for processing one video path into one frame
        single_video_to_frame_pipeline = (
            ToAbsolutePath(base_path) >>
            LoadVideo(
                num_frames=1,
                frame_processor=ImageCropAndResize(
                    height, width, max_pixels,
                    height_division_factor, width_division_factor
                )
            ) >>
            SelectFirstItem() >>
            ToTensor()
        )
        
        # Use RouteByType to handle both a single path (str) and a list of paths,
        # exactly like default_image_operator does.
        return RouteByType(operator_map=[
            (str, single_video_to_frame_pipeline),
            (list, SequencialProcess(single_video_to_frame_pipeline)),
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

    def _locate_transition(self, index):
        # find a particular data point within an episode
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index)  # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        return episode_index, start_ts

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
                        if key == 'vace_reference_image':
                           first_image_op = UnifiedDataset.default_first_image_operator()
                           data[key] = first_image_op(original_video_path)
                        elif key == 'video':
                           data[key] = self.main_data_operator(original_video_path)
                           self.num_frame = len(data[key])
                        elif key == 'prompt':
                           data[key] = data.get(key)
                        elif key == 'pose':
                           #self.dataset_path_list, self.episode_len = index_episodes(data['pose'])
                           #self.cumulative_len = np.cumsum(self.episode_len)
                           #episode_id, frame_id = self._locate_transition(data_id)
                           #hdf5_file = dataset_path_list[episode_id]
                           hdf5_file = data.get(key)
                           query_tfs = WRISTS
                           with h5py.File(hdf5_file, "r") as root:
                               available_cam_ext = root['/transforms/camera'][:]
                               num_available_frames = available_cam_ext.shape[0]

                               if num_available_frames == 0:
                                  raise ValueError(f"HDF5 file {hdf5_file} contains no frames.")
                               indices = np.arange(self.num_frame) % num_available_frames
                    
                               tfdtype = available_cam_ext.dtype
                               tfs = np.zeros([len(query_tfs), self.num_frame, 4, 4], dtype=tfdtype)
                               for i, tf_name in enumerate(query_tfs):
                                   available_tfs = root['/transforms/' + tf_name][:]
                                   tfs[i] = available_tfs[indices]

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
                               data[key] = {'tfs': tfs, 'cam_ext': cam_ext, 'cam_int': cam_int, 'lang_instruct': lang_instruct, 'confs': confs}

                                
                      
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
