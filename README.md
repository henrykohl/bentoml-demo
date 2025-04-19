# Bentoml-Demo

> 原 Lecture 出處 [Model Trainer & Bento ML](https://www.youtube.com/watch?v=Aahc28-f4hc) (57:00--1:23:00)
>
> 原 Github 出處 [BentoML demo repo](https://github.com/entbappy/bentoml-demo/)

- 此 project 實做是依據 **Gitpod** (而非 **Github Codespaces**)
  > 在 **Gitpod** 開啟 terminal 後，後續實做如下:

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
> > 執行命令: `bentoml serve service.py:service --reload` (演 1)
>
> `bentofile.yaml` (1:14:29)
>
> 執行命令: `bentoml build`

- 關於 " 演 1 "

> 開啟 Browser ，在 Service APIs 中 POST /predict 下的 Request body 輸入
>
> > [
> >
> > [2,3,4,5]
> >
> > ]
>
> 按下 Execute 後，在 Response body 中顯示
>
> > [
> >
> > 2
> >
> > ]

# BentoML Tutorial: Build Production Grade AI Applications 

[Lecture video](https://www.youtube.com/watch?v=i_FtfdOKa2M)

* 建立  `requirements.txt`
```sh
bentoml==1.0.25
scikit-learn
```

* 執行
```bash
conda create -p venv python==3.9 -y

conda activate venv

pip install -r requirements.txt
```

* 建立  `train.py`
```python
from sklearn import svm
from sklearn import datasets
import bentoml

# Load training data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Model Training
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

# Save model to the BentoML local model store
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print(f"Model saved: {saved_model}")
```

* 執行
```bash
python train.py

bentoml models list
```

* 建立 `test.py`
```python
import bentoml

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

iris_clf_runner.init_local()

print(iris_clf_runner.predict.run([[5.9, 3., 5.1, 1.8]]))  # => array(2)
```

* 執行
```bash
python test.py
```

* 建立 `service.py`
```python
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

# svc: Service model name
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = iris_clf_runner.predict.run(input_series)
    return result
```

* 執行
```bash
bentoml serve service.py:svc --reload
```

* 打開 browser 用網址 localhost:3000 在 POST/classify, 按`Try it out` 輸入 [[1,2.3,4.2,1.0]], 按`Execute` 
> 結果是  1

* 建立 `bentofile.yaml`
```sh
service: "service.py:svc"  
labels:
    owner: bentoml-team
    stage: demo
include:
 - "*.py" 
python:
  packages:
   - scikit-learn
   - pandas
```

* 執行
```bash
bentoml build
```

* 執行
```bash
bentoml list
```





















# 參考/補充

> Github 用 Gitpod 開啟時，若有 _requirements.txt_ 存在，會直接安裝在工作目錄，但我們希望建立 virtual environment 後，才將 _requirements.txt_ 安裝於其中，因此 BentoML Demo Repo 中預先不含 _requirements.txt_。

- 參 1 [Codespaces in GitHub](https://levelup.gitconnected.com/codespaces-in-github-6457533fc7f1)

- 參 2 [如何使用 GitPod](https://henrykohl-bentomldemo-sry846dwcvs.ws-us118.gitpod.io/)

- 補 0 **Gitpod** 安裝 conda，主要參考了 `參 1`

* 補 1

> `bentomldemo.ipynb` 是將此 project 改成用 Colab 運行，但無法成功

- 補 2

> `sklearn-sentiment-analysis.ipynb` 是一個使用 BentoML 的範例，可以成功地在 Colab 上運行

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
