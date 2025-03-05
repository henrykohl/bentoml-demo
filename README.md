# bentoml-demo
BentoML Demo

* requirements.txt
```
scikit-learn
bentoml==1.0.25
cattrs==23.1.1
```

* .gitpod.yml
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