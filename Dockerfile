FROM nvcr.io/nvidia/pytorch:22.05-py3

LABEL maintainer="Aleksa Sukovic" \
    project="LLDQN"

# Set environment variables.
#     - PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc.
#     - PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV HOME=/home/app

# System configuration and dependencies.
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    vim;
RUN python -m ensurepip --upgrade

# Dependencies.
COPY requirements.txt ${HOME}/
RUN pip install -r ${HOME}/requirements.txt
RUN rm ${HOME}/requirements.txt

ENTRYPOINT ["tail", "-f", "/dev/null"]
