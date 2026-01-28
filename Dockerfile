# NVIDIA GPU対応
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# 対話型プロンプトを無効化
ENV DEBIAN_FRONTEND=noninteractive

# 必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# pipのアップグレードとビルドツールの固定 (Gymビルドエラー回避のため)
RUN pip3 install --no-cache-dir "pip<24.1" && \
    pip3 install --no-cache-dir setuptools==65.5.0 wheel==0.38.4

# 強化学習関連と可視化ツールのインストール
RUN pip3 install --no-cache-dir \
    gym==0.23.1 \
    stable-baselines3[extra] \
    shimmy \
    pyyaml \
    imageio \
    imageio-ffmpeg \
    matplotlib

# f1tenth_gym のインストール
# /opt/f1tenth_gym にクローンして編集可能モードでインストール
RUN git clone https://github.com/f1tenth/f1tenth_gym.git /opt/f1tenth_gym
# 自作マップをビルド時に中に入れてしまう
COPY ./my_maps/my_map.pgm /opt/f1tenth_gym/gym/f110_gym/envs/maps/
COPY ./my_maps/my_map.yaml /opt/f1tenth_gym/gym/f110_gym/envs/maps/
WORKDIR /opt/f1tenth_gym
# setup.py の依存関係を無視してインストール (すでに gym 0.23.1 を導入済みのため)
RUN pip3 install -e . --no-deps

# マップファイルの準備（シミュレーターの仕様に合わせる）
RUN cp /opt/f1tenth_gym/gym/f110_gym/envs/maps/levine.pgm /opt/f1tenth_gym/gym/f110_gym/envs/maps/levine.png

# 作業ディレクトリに戻す
WORKDIR /workspace