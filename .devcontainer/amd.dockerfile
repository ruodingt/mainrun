FROM rocm/pytorch:latest

WORKDIR /home/ubuntu/workspace

RUN curl -1sLf 'https://dl.cloudsmith.io/public/task/task/setup.deb.sh' | sudo -E bash

RUN apt-get update && apt-get install -y \
    git \
    vim \
    curl \
    sudo \
    task \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /var/run/sshd

# Setup python venv
RUN python3 -m venv /opt/venv && chown -R ubuntu:ubuntu /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN usermod -aG sudo ubuntu && \
    usermod -aG video ubuntu && \
    usermod -aG render ubuntu && \
    usermod -aG kvm ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p /home/ubuntu/.cache && chown -R ubuntu:ubuntu /home/ubuntu/.cache

USER ubuntu
WORKDIR /home/ubuntu/workspace

RUN pip install requests pysocks datasets tokenizers structlog PyYAML tensorboard --proxy="" -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN echo "source /opt/venv/bin/activate" >> /home/ubuntu/.bashrc
RUN echo "export HSA_OVERRIDE_GFX_VERSION=11.5.1" >> /home/ubuntu/.bashrc
RUN echo "export PYTORCH_ROCM_ARCH=gfx1151" >> /home/ubuntu/.bashrc

# Environment overrides for Strix Halo (gfx1151)
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1
ENV PYTORCH_ROCM_ARCH=gfx1151
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["/bin/bash"]
