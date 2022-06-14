FROM python:3.9

LABEL maintainer="Aleksa Sukovic" \
    project="LLDQN"

# Set environment variables.
#     - PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc.
#     - PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive
ENV HOME=/home/app

# System configuration and dependencies.
RUN apt-get update && apt-get install --no-install-recommends -y \
    gcc \
    make \
    libnuma-dev \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    vim;
RUN python -m ensurepip --upgrade

# Dependencies.
COPY requirements.txt ${HOME}/
RUN pip install -r ${HOME}/requirements.txt
RUN rm ${HOME}/requirements.txt

WORKDIR ${HOME}
# Set command in Gradient CMS panel:
# jupyter notebook --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.trust_xheaders=True --NotebookApp.disable_check_xsrf=False --NotebookApp.allow_remote_access=True --NotebookApp.allow_origin='*'
