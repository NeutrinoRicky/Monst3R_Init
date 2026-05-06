LD_PRELOAD=/mnt/store/fd/project/DynamicReconstruction/monst3r/reflow_a1/libittnotify.so \
CUDA_VISIBLE_DEVICES=0 python \
/mnt/store/fd/project/DynamicReconstruction/monst3r/demo.py \
  --scene_root /mnt/store/fd/project/dataset/HyperNeRF/vrig/chicken-single \
  --scene_split train \
  --scene_camera_mode fixed \
  --scene_keyframe_stride 5 \
  --output_dir /mnt/store/fd/project/DynamicReconstruction/monst3r/demo/chicken

LD_PRELOAD=/mnt/store/fd/project/DynamicReconstruction/monst3r/reflow_a1/libittnotify.so \
CUDA_VISIBLE_DEVICES=1 /mnt/store/fd/app/anaconda3/envs/monst3r/bin/python \
/mnt/store/fd/project/DynamicReconstruction/monst3r/demo.py \
  --scene_root /mnt/store/fd/project/dataset/HyperNeRF/interp/cut-lemon1 \
  --scene_split train \
  --scene_camera_mode fixed \
  --scene_keyframe_stride 10 \
  --output_dir /mnt/store/fd/project/DynamicReconstruction/monst3r/demo/lemon

LD_PRELOAD=/mnt/store/fd/project/DynamicReconstruction/monst3r/reflow_a1/libittnotify.so \
CUDA_VISIBLE_DEVICES=1 /mnt/store/fd/app/anaconda3/envs/monst3r/bin/python \
/mnt/store/fd/project/DynamicReconstruction/monst3r/demo.py \
  --scene_root /mnt/store/fd/project/dataset/HyperNeRF/interp/torchocolate \
  --scene_split train \
  --scene_camera_mode fixed \
  --scene_keyframe_stride 10 \
  --output_dir /mnt/store/fd/project/DynamicReconstruction/monst3r/demo/torchocolate

LD_PRELOAD=/mnt/store/fd/project/DynamicReconstruction/monst3r/reflow_a1/libittnotify.so \
CUDA_VISIBLE_DEVICES=2 /mnt/store/fd/app/anaconda3/envs/monst3r/bin/python \
/mnt/store/fd/project/DynamicReconstruction/monst3r/demo.py \
--input_dir /mnt/store/fd/project/dataset/Nvidia_monocular/DynamicFace/images \
--scene_keyframe_stride 5 \
--scene_camera_mode fixed \
--output_dir /mnt/store/fd/project/DynamicReconstruction/monst3r/demo/DynamicFace

LD_PRELOAD=/mnt/store/fd/project/DynamicReconstruction/monst3r/reflow_a1/libittnotify.so \
CUDA_VISIBLE_DEVICES=2 /mnt/store/fd/app/anaconda3/envs/monst3r/bin/python \
/mnt/store/fd/project/DynamicReconstruction/monst3r/demo.py \
--scene_root /mnt/store/fd/project/dataset/Nvidia_monocular/Balloon1 \
--scene_camera_mode fixed \
--scene_keyframe_stride 5 \
--output_dir /mnt/store/fd/project/DynamicReconstruction/monst3r/demo/Balloon1

LD_PRELOAD=/mnt/store/fd/project/DynamicReconstruction/monst3r/reflow_a1/libittnotify.so \
CUDA_VISIBLE_DEVICES=2 /mnt/store/fd/app/anaconda3/envs/monst3r/bin/python \
/mnt/store/fd/project/DynamicReconstruction/monst3r/demo.py \
--scene_root /mnt/store/fd/project/dataset/Nvidia_monocular/Umbrella \
--scene_camera_mode fixed \
--scene_keyframe_stride 5 \
--output_dir /mnt/store/fd/project/dataset/Nvidia_monocular/Umbrella/images_2