# classifiers

** First time installation **

We need CUDA 10! Here for Ubuntu 18.04 https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal

```
conda env create -f env.yml

git submodule init
git submodule update
cd externals/detectron2 && pip install -e .
```


