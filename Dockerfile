# # NVIDIA GPU対応のROS2イメージをベースにする
# FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# # 環境変数の設定（インストール中の対話型プロンプトを避ける）
# ENV DEBIAN_FRONTEND=noninteractive

# # 必要なツールとPythonのインストール
# RUN apt-get update && apt-get install -y \
#     python3-pip python3-dev git libgl1-mesa-glx \
#     && rm -rf /var/lib/apt/lists/*

# # pipのバージョンを古いライブラリ(gym==0.19.0)が許容される24.0に固定
# RUN pip3 install "pip<24.1"

# # 強化学習ライブラリのインストール
# RUN pip3 install gym==0.19.0 stable-baselines3 shimmy gymnasium imageio imageio-ffmpeg

# # f1tenth_gym本体をインストール（コンテナ内の /opt に入れる）
# RUN git clone https://github.com/f1tenth/f1tenth_gym.git /opt/f1tenth_gym
# RUN cd /opt/f1tenth_gym && pip3 install -e .

# RUN cp /opt/f1tenth_gym/gym/f110_gym/envs/maps/levine.pgm /opt/f1tenth_gym/gym/f110_gym/envs/maps/levine.png

# WORKDIR /workspace




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

# pipのアップグレードとバージョン固定
RUN pip3 install --no-cache-dir "pip<24.1"

# 強化学習関連のインストール (Gym 0.19.0 を基準に SB3 を構成)
# ※ shimmy は Gymnasium 互換用ですが、最新の SB3 を入れるために同梱
RUN pip3 install --no-cache-dir \
    gym==0.19.0 \
    stable-baselines3[extra] \
    shimmy \
    gymnasium \
    imageio \
    imageio-ffmpeg

# f1tenth_gym のインストール
# /opt/f1tenth_gym にクローンして編集可能モードでインストール
RUN git clone https://github.com/f1tenth/f1tenth_gym.git /opt/f1tenth_gym
WORKDIR /opt/f1tenth_gym
RUN pip3 install -e .

# マップファイルの準備（シミュレーターの仕様に合わせる）
RUN cp /opt/f1tenth_gym/gym/f110_gym/envs/maps/levine.pgm /opt/f1tenth_gym/gym/f110_gym/envs/maps/levine.png

# 作業ディレクトリに戻す
WORKDIR /workspace