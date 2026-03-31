# LOME: Learning Human-Object Manipulation with Action-Conditioned Egocentric World Model

<p align="center">  
    <a href="https://zerg-overmind.github.io/">Quankai Gao</a><sup>1</sup>,
    <a href="https://jiawei-yang.github.io/">Jiawei Yang</a><sup>1</sup>,
    <a href="https://xharlie.github.io/">Qiangeng Xu</a><sup>2</sup>,
    <a href="https://clthegoat.github.io/">Le Chen</a><sup>3</sup>,
    <a href="https://yuewang.xyz/">Yue Wang</a><sup>1</sup>
    <br>
    <sup>1</sup>University of Southern California <sup>2</sup> Waymo <sup>3</sup> Max Planck Institute for Intelligent Syetems
</p>
<p align="center">
  <img src="/images/USC-Logos.png?raw=true" height="50" />
  <img src="/images/waymo_logo.png?raw=true" height="50" />
  <img src="/images/mpi_logo.png?raw=true" height="50" />
</p>

<div align="center">
    <a href="https://zerg-overmind.github.io/LOME.github.io/"><strong>Project Page</strong></a> |
    <a href="https://arxiv.org/abs/2603.27449"><strong>Paper</strong></a> |
    <a href="https://www.youtube.com/watch?v=zCBdu0Hvty4"><strong>Video</strong></a> 
</div>

## 1. Installation  
Our training implementation follows [Diffsynth-studio](https://github.com/modelscope/diffsynth-studio). 
```bash
git clone https://github.com/Zerg-Overmind/LOME.git --recursive
cd DiffSynth-Studio
pip install -e .
```
## 2. Training
Our base model is Wan2.1-VACE 14B and please refer to `job_ego_dex.sh` and `Wan2.1-VACE-14B_torch.sh` as an example for launching the multi-node training job on your cluster (slurm for example). Be sure to modify the slurm account, partition, and image path according to your cluster configuration. If you only want to try on single node, feel free to refer to `local_wan_torch_try.sh`. Pretrained checkpoints of diffusion module `diffusion_pytorch_model.safetensors`, Wan VAE `Wan2.1_VAE.pth` and text encoder `models_t5_umt5-xxl-enc-bf16.pth` should be downloaded automatically to folder `pretrained` after launching the training or inference job. Our checkpoint finetuned on [egodex dataset](https://github.com/apple/ml-egodex) can be downloaded to `models/train/Wan2.1-VACE-14B_full` from [Google Drive](https://drive.google.com/file/d/16DcayHS8oUZ014scwXvdgAY44xNeUEc2/view?usp=sharing), which corresponds to [Line 397](https://github.com/Zerg-Overmind/LOME/blob/main/diffsynth/pipelines/wan_video_new.py#L397) in `wan_video_new.py`.
You might want to download the [egodex dataset](https://github.com/apple/ml-egodex) dataset and preprocess it with `make_metadata_vace_filter.py` before training. Details of the dataset definition can be found at `diffsynth/trainers/naive_dataset.py`.
After generating 2D action maps with `make_metadata_vace_filter.py`, EgoDex dataset is as follows:
```
egodex
├── train (including all parts)
│   ├── <task_1>
│   │   ├── 0.hdf5
│   │   ├── 0.mp4
│   │   ├── 0.jpg
│   │   ├── 0_pose.mp4 (2D action map)
│   │   ├── 1.hdf5
│   │   └── 1.mp4
│   │   ...
│   ├── <task_2>
│   │   ├── 0.hdf5
│   │   ├── 0.mp4
│   │   ├── 0.jpg
│   │   ├── 0_pose.mp4 (2D action map)
│   │   ├── 1.hdf5
│   │   └── 1.mp4
│   │   ...
│   ...
├── test
│   ├── <task_n>
│   │   ├── 0.hdf5
│   │   ├── 0.mp4
│   │   ├── 0.jpg
│   │   ├── 0_pose.mp4 (2D action map)
│   │   ├── 1.hdf5
│   │   └── 1.mp4
│   │   ...
│   ...
```
Before launching the training, please make sure Line 1306-1311 in `wan_video_new.py` is not commented out and Line 1314-1318 is commented out.
## 3. Inference
Make sure you have downloaded the finetuned checkpoint `step-80_concat_new.safetensors` to `models/train/Wan2.1-VACE-14B_full`, and all other pretrained checkpoints including `diffusion_pytorch_model.safetensors`, `Wan2.1_VAE.pth` and `models_t5_umt5-xxl-enc-bf16.pth` under `pretrained`. 
Before inference, please comment out the code Line 1306-1311 in `wan_video_new.py`  and uncomment Line 1314-1318.
And then run: 
```bash
python inference_script_vace.py
```

## Acknowledgments
Our LOME is benefited from the following open-source projects:
- [Diffsynth-studio](https://github.com/modelscope/diffsynth-studio)
- [egodex dataset](https://github.com/apple/ml-egodex)
- [Wan2.1-VACE](https://github.com/ali-vilab/VACE)
- [Go-with-the-flow](https://github.com/gowiththeflowpaper/gowiththeflowpaper.github.io)
- [CoSHAND](https://github.com/SruthiSudhakar/CosHand)

## Citation
If you find our work useful in your research, please consider citing:
```bibtex
@article{gao2024lome,
  title={LOME: Learning Human-Object Manipulation with Action-Conditioned Egocentric World Model},
  author={Gao, Quankai and Yang, Jiawei and Xu, Qiangeng and Chen, Le and Wang, Yue},
  journal={arXiv preprint arXiv:},
  year={2024}
}