# Common IR

## 环境搭建

参考[FlagTree Wiki](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-ascend?open_in_browser=true)，以下方为准

### flagtree镜像

#### 910B

```bash
# 910B Plan A: docker pull (13.3GB)
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64:202606-torch2.9.0-base
docker pull ${IMAGE}
# 910B Plan B: docker load (4.8GB)
IMAGE=flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64:202606-torch2.9.0-base
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64.202606-torch2.9.0-base.tar.gz
docker load -i flagtree-ascend3.5-910b-py311-cann9.0.0-ubuntu22.04-aarch64.202606-torch2.9.0-base.tar.gz
```

#### 910C

```bash
# 910C Plan A: docker pull (19.2GB)
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64:202608-torch2.10.0-vllm0.20.2
docker pull ${IMAGE}
# 910C Plan B: docker load (6.6GB)
IMAGE=flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64:202608-torch2.10.0-vllm0.20.2
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64.202608-torch2.10.0-vllm0.20.2.tar.gz
docker load -i flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64.202608-torch2.10.0-vllm0.20.2.tar.gz
CONTAINER=flagtree-dev-xxx
docker run -dit -u 0 --user=root \
    --network=host --pid=host --ipc=host --privileged \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /etc/localtime:/etc/localtime:ro \
    -v /data:/data -v /home:/home \
    -w /root --name ${CONTAINER} ${IMAGE} bash
docker exec -it ${CONTAINER} /bin/bash
```

### 环境准备

#### CANN

安装 CANN 社区版9.1.0-昇腾社区 (同时需要 CANN Toolkit 和 ops 算子包)
参考: [CANN文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/softwareinst/instg/instg_0090.html?OS=Ubuntu&InstallType=localpack)

```bash
mkdir pkg
cd pkg

wget https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann_9.1.0_linux-aarch64.run
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-910b-ops_9.1.0_linux-aarch64.run

# xxx -> personal path
bash ./Ascend-cann_9.1.0_linux-aarch64.run --install --install-path=/data/jianheng/Ascend/910
# 910b
bash ./Ascend-cann-910b-ops_9.1.0_linux-aarch64.run --install --install-path=/data/jianheng/Ascend/910

source /data/jianheng/Ascend/910/ascend-toolkit/set_env.sh
source /data/jianheng/Ascend/910/cann/set_env.sh
```

#### LLVM

```bash
mkdir -p ~/.flagtree/ascend;
cd ~/.flagtree/ascend
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/llvm-7d5de303-ubuntu-aarch64-python311-compat_v0.6.0.tar.gz
tar zxvf llvm-7d5de303-ubuntu-aarch64-python311-compat_v0.6.0.tar.gz
```

#### Triton依赖

```bash
# For Triton 3.5 (aarch64)
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/build-deps-triton_3.5.x-linux-aarch64.tar.gz
sh python/scripts/unpack_triton_build_deps.sh ./build-deps-triton_3.5.x-linux-aarch64.tar.gz
```

### 源码构建步骤

```bash
git clone https://github.com/kernelllm/flagtree --branch common-ir-feature-tensor-view
cd flagtree
git checkout common-ir-feature-tensor-view
# 注意 third_party/flir 需要切换到同名分支，目前构建时会自动克隆并切换到对应提交
# 但如果构建时该目录已存在则不会自动处理
# Set FLAGTREE_BACKEND using the backend name from the table above
export FLAGTREE_BACKEND=ascend  # Do not set it on nvidia/amd/triton-shared
export MAX_JOBS=32
export LLVM_SYSPATH=~/.flagtree/ascend/llvm-7d5de303-ubuntu-aarch64-python311-compat
export PATH=~/.flagtree/ascend/llvm-7d5de303-ubuntu-aarch64-python311-compat/bin/:$PATH
# 安装
python -m pip install -r ./python/requirements-ascend.txt
python -m pip install . -v --no-build-isolation
# 验证算子
python test/CommonIR/Ascend/success_case/native_matmul.py
USE_CUSTOM_COMPILE_OPT=1 python test/CommonIR/Ascend/success_case/matmul_add_residual_cv.py
```

### 自定义run包

目前部分特性依赖华为提供的自定义AscendNPU-IR CANN包，目前没有公开的下载链接，请[点击入群](https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=536p0a6d-691b-42f2-8a52-477930161716)获取


