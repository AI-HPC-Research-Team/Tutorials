# NVIDIA&CUDA Environment Setup

- **Update source**
    ``` shell
    vim /etc/apt/sources.list 
    sudo apt-get update
    ```
- **Update host environment**
    ``` shell
    vim /etc/resolv.conf
    # nameserver 8.8.8.8
    /etc/init.d/networking restart
    apt update && apt install cmake make openssh-client openssh-server sshpass -y
    dpkg-reconfigure tzdata
    export LANG=C.UTF-8
    ```
- **Install Nvidia driver**
    ``` shell
    sudo apt-get install nvidia-driver-xxx
    sudo reboot
    nvidia-smi
    ```
- **Install CUDA**
    ``` shell
    # cuda toolkit
    sudo apt install nvidia-cuda-toolkit
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda-repo-ubuntu1804-11-4-local_11.4.1-470.57.02-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1804-11-4-local_11.4.1-470.57.02-1_amd64.deb
    sudo apt-key add /var/cuda-repo-ubuntu1804-11-4-local/7fa2af80.pub
    sudo apt-get update
    sudo apt-get -y install cuda
    nvcc -V 	# verify cuda version

    # cudnn lib
    tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
    sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
    sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
    sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
    sudo dpkg -i cudnn-local-repo-${OS}-8.x.x.x_1.0-1_amd64.deb
    sudo apt-get install libfreeimage3 libfreeimage-dev
    ```
- **Install docker**
    ``` shell
    sudo apt install curl
    curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
    docker
    install nvidia-docker2
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
    sudo apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    # nvidia-container-runtime
    sudo apt-get install -y nvidia-docker2
    sudo pkill -SIGHUP dockerd
    systemctl restart docker
    systemctl daemon-reload
    cat /etc/docker/daemon.json 
    sudo docker ps -a
    ```


