FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

USER root

# タイムゾーンの設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# プロキシの設定
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY

# 作業ディレクトリの作成
RUN mkdir -p /root/src
WORKDIR /root/src

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    zip unzip git curl tmux valgrind ffmpeg

# git-lfsのインストール
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && git lfs install

# LD_LIBRARY_PATHの設定
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# CUDAとNVIDIA Toolkitの設定
ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Pythonパッケージのインストール
RUN pip install --upgrade pip
RUN pip install openai-whisper pyannote.audio python-dotenv

# デフォルトのコマンド
CMD ["/bin/bash"]
