# PYTHON Environment Setup

- **Build container**
    ``` shell
    sudo docker pull nvcr.io/nvidia/pytorch:21.10-py3
    sudo docker run --runtime=nvidia --gpus all --net host -it -d --name test nvcr.io/nvidia/pytorch:21.09-py3
    # exit
    sudo docker exec it test bash
    dpkg-reconfigure tzdata
    vim /etc/bash.bashrc
    # export LANG=C.UTF-8
    source /etc/bash.bashrc
    which python
    nvidia-smi
    ```
- **Install conda**
    ``` shell
    wget -4 -c http://mirrors.aliyun.com/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    source ~/miniconda3/bin/activate
    conda init
    ```
- **Create virtual environment**
    ``` shell
    conda create -n test python=3.9
    conda activate test
    ```
- **Install libraries**
    ``` shell
    sudo apt install python3-dev python3-pip python3-venv
    # https://pytorch.org/get-started/locally/
    conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install --upgrade tensorflow
    pip install transformers
    which python
    pip list
    conda list
    ```

