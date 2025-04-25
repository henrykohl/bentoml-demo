# Bentoml-Demo (Notebook E1)

> 原 Lecture 出處 [Model Trainer & Bento ML](https://www.youtube.com/watch?v=Aahc28-f4hc) (57:00--1:23:00)
>
> 原 Github 出處 [BentoML demo repo](https://github.com/entbappy/bentoml-demo/)

- 此 project 實做是依據 **Gitpod** (而非 **Github Codespaces**)
  > 在 **Gitpod** 開啟 terminal 後，後續實做如下:

- 安裝 Conda

```bash
mkdir .tmp
cd .tmp
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh
```

* 開啟新的 bash shell，檢測 conda 是否安裝成功，若沒有用新的 shell 來執行會出現「bash: conda: command not found」

```bash
conda list
```

* 建立 virtual environment

```bash
conda create -p env python==3.8 -y
conda create --name env python==3.8 -y ## 另法
# conda create -p env python=3.8 -y ## 當啟動 env 後，python 版本依然是最新版，非版本3.8
```

* 啟動 env

```bash
conda activate env

python --version # 檢查 env 中 python 版本
```
* 建立 requirements.txt

```sh
scikit-learn
bentoml==1.0.25
cattrs==23.1.1
```

* 安裝 packages: 執行命令
```bash
pip install -r requirements.txt
```

* 建立 `bento_train.py`
```python
import bentoml

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
iris = load_iris()
X = iris.data[:, :4]
Y = iris.target
model.fit(X, Y)

bento_model = bentoml.sklearn.save_model('kneighbors', model)
print(f"Model saved: {bento_model}")
```

*  訓練模型: 執行命令
```bash
python bento_train.py
```

* 顯示模型: 執行命令
```bash
bentoml models list
```


* 建立  `bento_test.py`
```python
import bentoml

clf = bentoml.sklearn.get('kneighbors:latest').to_runner()
clf.init_local()
print(clf.predict.run([[2,3,4,5]]))
```


* 測試模型: 執行命令
```bash
python bento_test.py
```



* 建立  `service.py` (1:10:18)
```python
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

clf = bentoml.sklearn.get('kneighbors:latest').to_runner()

service = bentoml.Service(
    "kneighbors", runners=[clf]
)

# Create an API function
@service.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_series: np.ndarray) -> np.ndarray:

    result = clf.predict.run(input_series)
    
    return result
```

* 啟動 serve: 執行命令 (演 1)
```bash
bentoml serve service.py:service --reload
```

* 建立 `bentofile.yaml` (1:14:29)
```yaml
service: "service.py:service"  # Same as the argument passed to `bentoml serve`
labels:
    owner: bentoml-team
    project: gallery
include:
- "*.py"  # A pattern for matching which files to include in the bento
python:
    packages:  # Additional pip packages required by the service
    - scikit-learn
    - pandas
```


* 打包模型的所有 source code: 執行命令:
```bash
bentoml build
```

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

* 顯示已打包模型: 執行命令:
```bash
bentoml list
```


# BentoML Tutorial: Build Production Grade AI Applications (Notebook E4)

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

# BentoML SageMaker deployment (Notebook E5)

* [Lecture video](https://www.youtube.com/watch?v=Zci_D4az9FU)
> [Lecture code repository](https://github.com/jankrepl/mildlyoverfitted/blob/master/mini_tutorials/bentoml/README.md)

* 執行
```bash
# conda create -p venv python==3.9 -y 似乎不行
```

* 在 **Gitpod** 中不使用 conda environment，直接使用其環境， python --version 為 3.12.9

* 建立 `requirements.txt`
```sh
bentoctl==0.4.0
# bentoml==1.1.9 # bentoml build 會出問題
bentoml==1.4.10
boto3==1.29.0
numpy==1.26.2
pydantic==2.5.1
pydantic_core==2.14.3
scikit-learn==1.3.2
```

* 執行
```bash
pip install -r requirements.txt
```

* 在 **Gitpod**  中，，直接使用其環境， cattrs --version 為 23.1.2

* 建立 `create_model.py`
```python
import bentoml

from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
X, y = iris.data, iris.target

clf = svm.SVC(gamma="scale")
clf.fit(X, y)

saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print(saved_model)
```

* 執行
```bash
python create_model.py

bentoml models list
```

* 建立 `service.py`
```python
from typing import Literal

import bentoml

from pydantic import BaseModel
from bentoml.io import JSON


iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

class Request(BaseModel):
    sepal_width: float
    sepal_length: float
    petal_width: float
    petal_length: float

class Response(BaseModel):
    label: Literal["setosa", "versicolor", "virginica"]


@svc.api(input=JSON(pydantic_model=Request), output=JSON(pydantic_model=Response))
def classify(request: Request) -> Response:
    input_ = [
        request.sepal_width,
        request.sepal_length,
        request.petal_width,
        request.petal_length,
    ]

    label_index = iris_clf_runner.predict.run([input_])[0]
    label = ["setosa", "versicolor", "virginica"][label_index]

    return Response(label=label)
```

* 執行
```bash
bentoml serve service.py
```

* 打開 browser 用網址 localhost:3000 在 POST/classify, 按`Try it out` 輸入 
> { \
>   "sepal_width": 7, \
>   "sepal_length": 7, \
>   "petal_width": 7, \
>   "petal_length": 7 \
> } 

* 按`Execute` 結果是  
> { \
>   "label": "virginica" \
> }

* 建立 `bentofile.yaml`
```sh
service: "service:svc"
include:
- "service.py"
python:
  packages:
  - pydantic
  - scikit-learn
models:
- iris_clf:latest
```

* 執行
```bash
bentoml build
```

* 執行
```bash
bentoml list
```

* Install SageMaker operator -- 執行 (18:37)
```bash
bentoctl operator install aws-sagemaker
```

* 執行
```bash
bentoctl init
```

name: great-iris

operator/name: aws-sagemaker

region: us-east-1

instance_type: m1.t2.medium

initial_instance_count: 1

timeout: 60

enable_data_capture: False

destination_s3_uri: 

initial_sampling_percentage: 1

Do you want to add item to env (y/n): n

filename for deployment_config [deployment_config.yaml]:

* 查看 `deployment_config.yaml`

* 查看 `bentoctl.tfvars`

* 查看 `main.tf`

* 安裝 `awscli` (Lecture 缺這一步驟)，否則下一步驟 `bentoctl build -f ...` 會出現 
> `botocore.exceptions.nocredentialserror: unable to locate credentials` 錯誤

* 執行
```bash
bentoctl build -f deployment_config.yaml -b iris_classifier:編號
```

* 再查看 `bentoctl.tfvars`

* 執行
```bash
aws ecr describe-repositories
```

* 執行
```bash
aws ecr list-images --repository-name=great-iris
```

* 安裝 `terraform` -- 執行 `brew install terraform`  (Lecture 缺這一步驟)

* 執行
```bash
terraform init

terraform plan -var-file=bentoctl.tfvars

terraform apply -var-file=bentoctl.tfvars
```
> 最後一步驟無法成功，出現 `error creating sagemaker endpoint: resourcelimitexceeded: resource limits for this account have been exceeded.` 錯誤，原因是 AWS 的流量限制，需要跟 AWS 客服做 request.
>
> 執行 `bentoctl init` 時，也許調整 **instance_type** 可以解決~~ 


* 執行
```bash
terraform destroy -var-file=bentoctl.tfvars # 並不會刪除 AWS 中的 repository

bentoctl destroy # 徹底刪除 AWS 中的 repository

aws sagemaker list-models # 查看

aws sagemaker list-endpoints # 查看
```




# 參考/補充

> Github 用 Gitpod 開啟時，若有 _requirements.txt_ 存在，會直接安裝在工作目錄，但我們希望建立 virtual environment 後，才將 _requirements.txt_ 安裝於其中，因此 BentoML Demo Repo 中預先不含 _requirements.txt_。

- 參 1 [Codespaces in GitHub](https://levelup.gitconnected.com/codespaces-in-github-6457533fc7f1)

- 參 2 [如何使用 GitPod](https://henrykohl-bentomldemo-sry846dwcvs.ws-us118.gitpod.io/)

- 補 0 
> **Gitpod** 安裝 conda，主要參考了 `參 1`

* 補 1

> `bentomldemo.ipynb` 是將此 project 改成用 Colab 運行，但無法成功

- 補 2

> `sklearn-sentiment-analysis.ipynb` 是一個使用 BentoML 的範例，可以成功地在 Colab 上運行

* 補 3
> `bentomlnotebook.ipynb` 共有 5 個範例，包含此 `README.md` 中的 3 個範例

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
