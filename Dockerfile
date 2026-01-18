# NVIDIA GPU対応のROS2イメージをベースにする
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# 必要なツールとPythonのインストール
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# f1tenth_gymと強化学習ライブラリのインストール
RUN pip3 install --upgrade pip
RUN pip3 install gym==0.19.0 stable-baselines3 shimmy gymnasium

# f1tenth_gym本体をインストール
RUN git clone https://github.com/f1tenth/f1tenth_gym.git /opt/f1tenth_gym
RUN cd /opt/f1tenth_gym && pip3 install -e .

WORKDIR /workspace