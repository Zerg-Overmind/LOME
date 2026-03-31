import os
import csv
import glob
import h5py
import cv2
from tqdm import tqdm # <-- 1. Import tqdm
from diffsynth.trainers.skeleton_tfs import WRISTS, NECK, DEFAULT_TFS
from diffsynth.trainers.skeleton_tfs import RIGHT_FINGERS, RIGHT_INDEX, RIGHT_THUMB, RIGHT_RING, RIGHT_MIDDLE, RIGHT_LITTLE
from diffsynth.trainers.skeleton_tfs import LEFT_FINGERS, LEFT_INDEX, LEFT_THUMB, LEFT_RING, LEFT_MIDDLE, LEFT_LITTLE
import numpy as np
import cv2
import imageio
import torch

QUERY_TFS = RIGHT_FINGERS + ['rightHand', 'rightForearm'] + LEFT_FINGERS + ['leftHand', 'leftForearm']
query_tfs = QUERY_TFS #DEFAULT_TFS #WRISTS

def is_video_valid_ffprobe(path):
    """Return True if ffprobe can parse video metadata."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,duration",
                "-of", "csv=p=0",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=2,
        )
        out = result.stdout.decode().strip()
        return bool(out) and "N/A" not in out and "0.0" not in out
    except Exception:
        return False

def process_nested_folders(root_folder):
    """
    Recursively scans for video/HDF5 pairs and creates a metadata.csv file,
    filtering out videos and extracting first frames with a progress bar.
    """
    fixed_prompt = " "
    def map_fingers_to_colors(tf_names):
        colors = []
        for tf in tf_names:
            if 'little' in tf.lower():
                colors.append((0, 152, 191))  # light blue
            elif 'ring' in tf.lower():
                colors.append((173, 255, 47))   # green yellow
            elif 'middle' in tf.lower():
                colors.append((230, 245, 250))  # pale torquoise
            elif 'index' in tf.lower():
                colors.append((255, 99, 71))    # tomato
            elif 'thumb' in tf.lower():
                colors.append((238, 130, 238))  # violet
        return np.array(colors)

    def imgs_to_mp4(img_list, mp4_path, fps=30):
        writer = imageio.get_writer(mp4_path, fps=fps, codec='libx264', format='FFMPEG', quality=8, pixelformat='yuv420p')
        for img in img_list:
            writer.append_data(img)  # img must be RGB (H,W,3) uint8
        writer.close()

    def draw_line(pointa, pointb, image, intrinsic, color=(0,255,0), thickness=5):
        # project 3d points into 2d
        pointa2, _ = cv2.projectPoints(pointa, np.eye(3), np.zeros(3), intrinsic, distCoeffs=np.zeros(5))  
        pointb2, _ = cv2.projectPoints(pointb, np.eye(3), np.zeros(3), intrinsic, distCoeffs=np.zeros(5))  

        # force to shape (2,)
        pointa2 = pointa2.reshape(-1)[:2]
        pointb2 = pointb2.reshape(-1)[:2]

        try:
            # convert to python ints
            pa = (int(pointa2[0]), int(pointa2[1]))
            pb = (int(pointb2[0]), int(pointb2[1]))

            # don't draw if completely out of bounds
            H, W, _ = image.shape
            if (pb[0] < 0 and pa[0] > W) or (pb[1] < 0 and pa[1] > H) or (pa[0] < 0 and pb[0] > W) or (pa[1] < 0 and pb[1] > H):
                return 

            # draw
            cv2.line(image, pa, pb, color=color, thickness=thickness)
            cv2.circle(image, pa, 5, color, -1)
            cv2.circle(image, pb, 5, color, -1)

        except (cv2.error, ValueError, TypeError) as e:
            # If an error occurs, print the diagnostic information
            print("\n--- Error Caught in draw_line ---")
            print(f"OpenCV Error: {e}")
            print(f"Point A data: {pointa2}, Shape: {pointa2.shape}, Type: {pointa2.dtype}")
            print(f"Point B data: {pointb2}, Shape: {pointb2.shape}, Type: {pointb2.dtype}")
            print("---------------------------------\n")

    def draw_line_sequence(points_list, image, intrinsic, color=(0,255,0)):
        # draw a sequence of lines in-place
        ptm = points_list[0]
        for pt in points_list[1:]:
            draw_line(ptm, pt, image, intrinsic, color)
            ptm = pt
            
    def convert_to_camera_frame(tfs, cam_ext):
        '''
        tfs: a set of transforms in the world frame, shape N x 4 x 4
        cam_ext: camera extrinsics in the world frame, shape 4 x 4
        '''
        return np.linalg.inv(cam_ext)[None] @ tfs

    def draw_hand(hand_dict, tfs_in_cam, cam_img, cam_int, right=True):
        # draw fingers
        for finger in ['little', 'ring', 'middle', 'index', 'thumb']: # roughly stack lines so closer fingers are drawn on top
            points = get_finger_pts(hand_dict[finger], tfs_in_cam, right)
            draw_line_sequence(points, cam_img, cam_int,color=map_fingers_to_colors([finger])[0].tolist())
        
        # draw forearm
        if right:
            forearm_points = [tfs_in_cam[tf2idx['rightForearm'], :3, -1]]
            forearm_points.append(tfs_in_cam[tf2idx['rightHand'], :3, -1])
        else:
            forearm_points = [tfs_in_cam[tf2idx['leftForearm'], :3, -1]]
            forearm_points.append(tfs_in_cam[tf2idx['leftHand'], :3, -1])
        draw_line_sequence(forearm_points, cam_img, cam_int,
                        color=map_fingers_to_colors(['middle'])[0].tolist())

    def get_finger_pts(finger_tf_names, tfs_in_cam, right=True):
        hand_name = 'rightHand'
        if not right:
            hand_name = 'leftHand'
        finger_points = [tfs_in_cam[tf2idx[hand_name], :3, -1]] # grab 3D position from SE(3) pose
        for tfname in finger_tf_names:
            finger_points.append(tfs_in_cam[tf2idx[tfname], :3, -1])
        return finger_points

    # Define a function to recursively print the structure
    def print_hdf5_structure(name, obj):
        """Prints the name and type of objects in an HDF5 file."""
        print(name, "(Group)" if isinstance(obj, h5py.Group) else f"(Dataset, shape: {obj.shape}, dtype: {obj.dtype})")

    # --- Filter thresholds ---
    MIN_DURATION_SECONDS = 0.5
    MIN_FRAME_COUNT = 2

    print(f"Recursively scanning all directories in '{root_folder}'...")
    # Convert generator to a list so we can slice
    all_dirs = list(os.walk(root_folder))
    total_dirs = len(all_dirs)
    start_index = 240
    print(f"📂 Found {total_dirs} folders total. Starting from index {start_index}.")
    
    for i, (dirpath, dirnames, filenames) in enumerate(all_dirs[start_index:], start=start_index):
        dirnames[:] = [d for d in dirnames if not d.endswith('.zip')]
        video_files = [f for f in filenames if f.endswith('.mp4') and not f.endswith('_pose.mp4')]
        
        if video_files:
            all_rows = []
            header = ['video', 'prompt', 'vace_video', 'vace_reference_image', 'pose']

            for video_filename in tqdm(video_files, desc=f"Processing {i/total_dirs} {os.path.basename(dirpath)}"):
                base_name = os.path.splitext(video_filename)[0]
                full_video_path = os.path.join(dirpath, video_filename)
                
                # --- START: Validation Logic ---
                hdf5_filename = f"{base_name}.hdf5"
                full_hdf5_path = os.path.join(dirpath, hdf5_filename)
                if not os.path.exists(full_hdf5_path):
                    continue

                try:
                    video = cv2.VideoCapture(full_video_path)
                    if not video.isOpened():
                        continue
                    
                    success, first_frame = video.read()
                    #######
                    aaaa=torch.tensor(first_frame).cuda()

                    #######
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = video.get(cv2.CAP_PROP_FPS)
                    video.release()
                    
                    if not success:
                        continue
                    
                    duration = frame_count / fps if fps > 0 else 0
                except Exception:
                    continue

                if frame_count < MIN_FRAME_COUNT or duration < MIN_DURATION_SECONDS:
                    continue
                
                ##############################################
                mp4_filename = f"{base_name}.mp4"
                pose_filname = f"{base_name}_pose.mp4"
                pose_path = os.path.join(dirpath, pose_filname)
                video_path = os.path.join(dirpath, mp4_filename)
                # skeleton visualization
                if not os.path.isfile(pose_path):
                    with h5py.File(full_hdf5_path, 'r') as f:
                        tfdtype = f['/transforms/camera'][0].dtype
                        tfs = np.zeros([len(query_tfs),frame_count, 4, 4], dtype=tfdtype)
                        for i, tf_name in enumerate(query_tfs):
                            tfs[i] = f['/transforms/' + tf_name][:frame_count]

                        cam_ext = f['/transforms/camera'][:frame_count] # extrinsics
                        cam_int = f['/camera/intrinsic'][:] # intrinsics

                        if f.attrs['llm_type'] == 'reversible':
                            direction = f.attrs['which_llm_description']
                            lang_instruct = f.attrs['llm_description' if direction == '1' else 'llm_description2']
                        else:
                            lang_instruct = f.attrs['llm_description']

                        confs = None
                        if 'confidences' in f.keys():
                            confs = np.zeros([len(query_tfs), frame_count],dtype=tfdtype)
                            for i, tf_name in enumerate(query_tfs):
                                confs[i] = f['/confidences/' + tf_name][:frame_count]
                
                        right_dict = {'index': RIGHT_INDEX, 'thumb': RIGHT_THUMB, 'middle': RIGHT_MIDDLE, 'ring': RIGHT_RING, 'little': RIGHT_LITTLE}
                        left_dict = {'index': LEFT_INDEX, 'thumb': LEFT_THUMB, 'middle': LEFT_MIDDLE, 'ring': LEFT_RING, 'little': LEFT_LITTLE}
                        tf2idx = {k: i for i, k in enumerate(QUERY_TFS)}

                        out_imgs = []
                        video = cv2.VideoCapture(video_path)
                        for num_f in range(frame_count):
                            ret, cam_img = video.read()
                            cam_img = np.zeros_like(cam_img)  # black bg
                            if not ret:
                                print("Error: Could not read frame.")
                                break
                            tfs_in_cam = convert_to_camera_frame(tfs[:, num_f,:,:], cam_ext[num_f])

                            draw_hand(right_dict, tfs_in_cam, cam_img, cam_int, right=True)
                            draw_hand(left_dict, tfs_in_cam, cam_img, cam_int, right=False)

                            out_imgs.append(cam_img)
                        imgs_to_mp4(out_imgs, pose_path, fps=fps)
                ##############################################
                # do not write in csv file if no pose file
                
                
                cap = cv2.VideoCapture(pose_path)
                ok, _ = cap.read()
                cap.release()
                # ok= True
                if ok: 
                    reference_image_filename = f"{base_name}.jpg"
                    output_image_path = os.path.join(dirpath, reference_image_filename)
                    if not os.path.exists(output_image_path):
                        cv2.imwrite(output_image_path, first_frame)
                    row = [video_filename, fixed_prompt, pose_filname, reference_image_filename, hdf5_filename]
                    all_rows.append(row)

            if all_rows:
                print(f"✅ Found {len(all_rows)} valid videos with pose. Creating CSV and reference images in '{dirpath}'...")
                output_csv_file = os.path.join(dirpath, "metadata_vace.csv")
                
                with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header)
                    writer.writerows(all_rows)
            else:
                print(f"No valid videos found in '{dirpath}' after filtering. No CSV or images created.")

    print("\nComplete.")


if __name__ == "__main__":
    data_path = "yours/datasets/ego-dex/"
    process_nested_folders(data_path)


