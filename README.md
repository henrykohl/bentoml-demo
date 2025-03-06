# Bentoml-Demo

> 原 Lecture 出處 [Model Trainer & Bento ML](https://www.youtube.com/watch?v=Aahc28-f4hc) (57:00--1:23:00)
>
> 原 Github 出處 [BentoML demo repo](https://github.com/entbappy/bentoml-demo/)

- 此 project 實做是依據 **Gitpod** (而非 **Github Codespaces**)

* 若不存在，則需先建立 requirements.txt

```
scikit-learn
bentoml==1.0.25
cattrs==23.1.1
```

- 安裝 Conda

```bash
mkdir .tmp
cd .tmp
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh
```

> 開啟新的 bash shell，檢測 conda 是否安裝成功，若沒有用新的 shell 來執行會出現「bash: conda: command not found」

```bash
conda list
```

> 建立 virtual environment

```bash
conda create -p env python==3.8 -y
conda create --name env python==3.8 -y ## 另法
# conda create -p env python=3.8 -y ## 當啟動 env 後，python 版本依然是最新版，非版本3.8
```

> 啟動 env

```bash
conda activate env
```

> 檢查 env 中 python 版本

```bash
python --version
# python -v  ## 不行
```

- Project workflow

> 執行命令: `conda activate env` (env 需要先建立)
>
> `requirements.txt`
>
> > 執行命令: `pip install -r requirements.txt`
>
> `bento_train.py`
>
> > 執行命令: `python bento_train.py`
>
> `bento_cmd.txt`
>
> `bento_test.py`
>
> > 執行命令: `python bento_test.py`
>
> `service.py` (1:10:18)
>
> `bentoml_cmd.txt`
>
> > 執行命令: `bentoml serve service.py:service --reload`
>
> `bentofile.yaml` (1:14:29)
>
> 執行命令: `bentoml build`

- 補充

> Github 用 Gitpod 開啟時，若有 _requirements.txt_ 存在，會直接安裝在工作目錄，但我們希望建立 virtual environment 後，才將 _requirements.txt_ 安裝於其中，因此 BentoML Demo Repo 中預先不含 _requirements.txt_。

# 忽略

- .gitpod.yml

```
tasks:

    - name: Mambaforge + dev.env setup

      init: |
        wget -O Mambaforge.sh  "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
        bash Mambaforge.sh -b -p "${HOME}/conda" && rm -f Mambaforge.sh
        source "${HOME}/conda/etc/profile.d/conda.sh"
        source "${HOME}/conda/etc/profile.d/mamba.sh"
        mamba env create -f environment.yml
```
