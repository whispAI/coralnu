### ~~Initial Server Setup~~
> - Launched AWS EC2 `t3.xlarge` instance
> - Connection via `coref.pem` file
    - Stored in S3 at [s3://whisp-research-keys/ensemble-coref/](https://whisp-research-keys.s3.eu-west-2.amazonaws.com/ensemble-coref/coref.pem)
> - `sudo apt-get update` and `sudo apt-get upgrade`
> - Install Anaconda
    - Make new env: `conda create --name coref python=3.6`
    - Updates `conda update conda --all` and `conda update anaconda`
    - Activate env `conda activate coref`
        - Install GitHub CLI `conda install gh --channel conda-forge`
            - Run `gh auth login` to authenticate and `gh auth setup-git`
            
### ~~Environment for Coref~~
> - Clone GH repo [NeuroSYS-pl/coreference-resolution](https://github.com/NeuroSYS-pl/coreference-resolution)
> - `conda activate coref`
> - install `gcc` and `make` with `sudo apt-get install make gcc`
    - also `sudo apt-get install python3-dev`

**Installation instructions for dependencies (stolen from OG repo)**
```
pip install spacy==2.1
python -m spacy download en_core_web_sm
pip install neuralcoref --no-binary neuralcoref
pip install allennlp
pip install --pre allennlp-models
```

@lucafrost — failing on install of spaCy due to C / Cython
- retry after running `conda install -c conda-forge gcc` failed as `cc1plus` fails to execute
    - as per [StackOverflow](https://stackoverflow.com/questions/69485181/how-to-install-g-on-conda-under-linux), retrying with `conda install -c conda-forge gxx` and `conda install -c conda-forge cxx-compiler`
    
~ 17/10/22

---
### Restarting efforts in Jupyter
- The EC2 instance continues not to cooperate in building spaCy, specifically a package called `preshed` — this issue appears to be caused by the installation (or lack thereof) of a C++ compiler for some relevant cython code.
    - I have tried installing every remedial solution I could find, including `gxx`, `cxx-compiler`, `python-dev`, etc...
    - While lazy, a managed environment makes the most sense, AWS will not have data science clients encountering C++/ObjC errors.
- Instantiated a `Python 3 (PyTorch 1.10 Python 3.8 CPU Optimized)` kernel image in AWS SageMaker Studio.
- Clone git repository [NeuroSYS-pl/coreference-resolution](https://github.com/NeuroSYS-pl/coreference-resolution)
- Successfully installed dependencies as below...
```console
pip install spacy==2.1
python -m spacy download en_core_web_sm
pip install neuralcoref --no-binary neuralcoref
pip install allennlp
pip install --pre allennlp-models
```
*\**I ran these in-notebook with `!pip` but I think an Image Terminal will also suffice*

right, now to get to the actual work...

~ 18/10/22 :: 13:44 GST

---

### updates
- coref resolution with spaCy neuralcoref is up and running, ran into an issue with the `Predictor` class in AllenNLP: missing package 'ipywidgets'
    - fix with `pip install ipywidgets` & restart kernel
- ran into issue with kernel death upon calling `predictor = Predictor.from_path(model_url)`
    - silly me, the instance only had 4GB of memory, upgrading to `ml.g4dn.xlarge`
- all done with both spaCy neuralcoref and AllenNLP pretrained SpanBERT. have used the intersection strategies implemented by @mmaslankowska-neurosys.
    - anecdotally, the `FuzzyIntersectionStrategy` appears to be the most effective.