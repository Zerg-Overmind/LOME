import torch
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.naive_dataset import UnifiedDataset
from tqdm import tqdm
from diffsynth.models import ModelManager
from diffsynth.pipelines.wan_video_new import WanVideoPipeline
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go


def vis_3d(frames):
    all_points = np.concatenate(frames, axis=0)  # (T*K, 3)

    xmin, ymin, zmin = all_points.min(axis=0)
    xmax, ymax, zmax = all_points.max(axis=0)

    # Add padding for better visualization
    pad = 0.05 * max(xmax - xmin, ymax - ymin, zmax - zmin)

    fig = go.Figure()

    def add_hand_keypoints(points, color):
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            name="joints",
            marker=dict(size=4, color=color, opacity=0.7)
        ))

    # Add initial frame
    add_hand_keypoints(frames[0], 'green')

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(range=[xmin - pad, xmax + pad]),
            yaxis=dict(range=[ymin - pad, ymax + pad]),
            zaxis=dict(range=[zmin - pad, zmax + pad]),
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
        ),
        title="3D Visualization",
        showlegend=False,
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {
                        'frame': {'duration': 50, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'type': 'buttons',
            'direction': 'left',
            'showactive': True
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Frame: '},
            'pad': {'t': 50},
            'len': 0.9,
            'x': 0.1,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top',
            'steps': [{
                'args': [[str(i)], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }],
                'label': str(i),
                'method': 'animate'
            } for i in range(len(frames))]
        }]
    )

    fig.frames = [
        go.Frame(
            data=[go.Scatter3d(
                x=frame[:, 0],
                y=frame[:, 1],
                z=frame[:, 2],
                mode='markers',
                marker=dict(size=4, color='green', opacity=0.7)
            )],
            name=str(i)
        )
        for i, frame in enumerate(frames)
    ]

    fig.show()
    return frames

MOTION_LABELS = [
                "Left", "Right", "Up", "Down",
                "LeftUp", "LeftDown", "RightUp", "RightDown",
                "In", "Out", "Static"
            ]

dataset_base_path = "yours/ego-dex-test/test"
dataset_metadata_path = dataset_base_path + "metadata_vace.csv"
data_file_keys = "video,prompt,vace_video,vace_reference_image,pose"
dataset = UnifiedDataset(
        base_path=dataset_base_path,
        metadata_path=dataset_metadata_path,
        repeat=1,
        data_file_keys=data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=dataset_base_path,
            max_pixels=480*832,
            height=480,
            width=832,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=81,
            time_division_factor=4,
            time_division_remainder=1,
        ),
)

train_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=lambda x: x[0],
        shuffle=False,
        num_workers=0
        )
model_manager = ModelManager()


device = 'cuda'
pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device=device),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device=device),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device=device),
        ],
)
main_data_operator=UnifiedDataset.default_video_operator(
            base_path=dataset_base_path,
            max_pixels=480*832,
            height=480,
            width=832,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=81,
            time_division_factor=4,
            time_division_remainder=1,
)
      
      
pipe.enable_vram_management()
for i, data in tqdm(enumerate(train_loader)):
      print(f'inference on instance:{data["pose"]["path"]}')
      print(f'prompt: {data["pose"]["lang_instruct"]}')
      video = pipe(
          prompt=data['pose']['lang_instruct'],
          camera_control_direction = data['camera_control_direction'],
          camera_control_speed = data['camera_control_speed'],
          pose = data['pose'],
          vace_video = data["vace_video"], # 2D action map
          reference_image = data["vace_reference_image"], 
          vace_reference_image = data["vace_reference_image"],
          negative_prompt="Bright colors, overexposed, static, blurry details, subtitles, style, artwork, image, still, overall gray, worst quality, low quality, JPEG compression artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, malformed limbs, fused fingers, still frame, messy background, three legs, crowded background people, walking backwards",
          seed=0, tiled=True,
      )
      save_video(video, f"output_examples/video_part2_default_cfg_{i}_test.mp4", fps=15, quality=5)
      save_video(data["video"], f"output_examples/original_video_{i}_test.mp4", fps=15, quality=5)
      save_video(data["vace_video"], f"output_examples/original_action_{i}_test.mp4", fps=15, quality=5)

      
