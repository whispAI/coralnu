<!--- WHISP DEVELOPMENT LOGO ~ RESPONSIVE TO LIGHT/DARK MODE --->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://i.imgur.com/xLIjgR0.png" height="37" align="right">
  <img align="right" src="https://i.imgur.com/aDti3wF.png" height="37">
</picture>
<br><br>

# coralnu 🪸
## coref resolution with spaCy `neuralcoref` + AllenNLP
> 💡 this repository stores the code for a flask inference wrapper for **coralnu** — a coreference resolution implementation that combines neuralcoref and AllenNLP — alongside scripts to deploy the code for inference using WSGI/Gunicorn.

### about coralnu
**coralnu** [cor-al-noo] performs **CO**reference **R**esolution with **Al**lennlp and **N**e**u**ralcoref using ensemble methods to achieve a fuzzy intersection. to combine the clusters identified by both spaCy `neuralcoref` and AllenNLP's `coref-spanbert-large`, coralnu uses a method of intersection that favours AllenNLP (owing to high GAP performance) as the ground truth, and includes all spans that partially overlap in `neuralcoref` and `AllenNLP` clusters, but prioritises the shorter span. Find out more in NeuroSYS's [blog post](https://neurosys.com/blog/effective-coreference-resolution-model) or browse the code.

### quickstart: make requests to the hosted endpoint
<img src="https://img.shields.io/badge/endpoint%20status-online-brightgreen">

**N.B.** There is an issue with the [coralnu.whisp.dev](https://coralnu.whisp.dev) endpoint, as the `nginx` proxy is refusing requests based on HTTP headers — to avoid this issue, please use the script below. Progress on this issue is being [tracked here](https://github.com/whispAI/coralnu/issues/2).

```python
import requests

def get_corefs(text):
    url = "https://coralnu.whisp.dev/coref"
    myobj = {'text': text}
    x = requests.post(url, json = myobj)
    return x.json()

text = "Born and raised in London, Daniel Day-Lewis excelled on stage at the National Youth Theatre, before being accepted at the Bristol Old Vic Theatre School, which he attended for three years."

get_corefs(text)
```
**Returns**
```json
{
  "clusters" : [
    [
      [
        6,
        9
      ],
      [
        31,
        31
      ]
    ]
  ],
  "method" : "fuzzyIntersection",
  "resolved" : "Born and raised in London, Daniel Day-Lewis excelled on stage at the National Youth Theatre, before being accepted at the Bristol Old Vic Theatre School, which Daniel Day-Lewis attended for 3 years"
}
```

### installation guide
**N.B.** this guide has been prepared and tested on Ubuntu 20.04 running on an AWS [g4dn.xlarge] instance, equipped with an NVIDIA T4 GPU.

**Initial Configuration**
* Perform one-time initial updates with `sudo apt-get update` and `sudo apt-get upgrade`
* Download the latest [Anaconda distribution](https://www.anaconda.com/products/distribution) installer, for example `curl -O https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh`.
  * Start the installation with `bash Anaconda3-2022.10-Linux-x86_64.sh` and follow the prompts
* Set up your `conda` environment with `conda create --name coralnu python=3.7`

**Install Dependencies**
* Activate your new conda environment with `conda activate coralnu`, you should now begin to install the dependencies.
* Install build tools as below:
  * `conda install -c conda-forge gxx`
  * `conda install -c conda-forge cxx-compiler`
  * `sudo apt-get install python3-dev`
* **Install AllenNLP:** first `‌pip install allennlp` then `pip install --pre allennlp-models`
* At this point, you may wish to check that CUDA is configured correctly with a simple script as below...
```py
import torch
x = torch.randn(10, 10, device='cuda')
print(x)
```
  * You should receive a tensor as an output.
* **Install Neuralcoref:** first, run `pip install spacy==2.1` to install the version of spaCy compatible with both coreference frameworks. Ensure to install a suitable spaCy model with `python -m spacy download en_core_web_sm`.
  * Finally, install `neuralcoref` with `pip install neuralcoref --no-binary neuralcoref`

**Prepare `flask` server**
* Install Flask with `pip install flask`
... will finish @lucafrost

## References

| Implementation | Description | License |
|----------------|-------------|---------|
[Huggingface NeuralCoref 4.0](https://github.com/huggingface/neuralcoref) | a CR extension for spaCy, `NeuralCoref` resolves coreference clusters using neural networks. More information can be found on their [blog post](https://medium.com/huggingface/state-of-the-art-neural-coreference-resolution-for-chatbots-3302365dcf30) and the [demo](https://huggingface.co/coref/). | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  <img width=250/> 
[AllenNLP coreference resolution](https://github.com/allenai/allennlp-models) | open-source project part of [AI2](https://allenai.org/) institute which introduced [ELMo](https://allennlp.org/elmo). Their span-ranking coreference resolution model is also premised upon a neural model. More information is provided by their [demo](https://demo.allennlp.org/coreference-resolution). | [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  <img width=250/> 
[NeuroSYS-pl/coreference-resolution](https://github.com/NeuroSYS-pl/coreference-resolution). | NeuroSYS implementation of 4 intersection methods on AllenNLP and Neuralcoref CR clusters. More information can be found on their [blog post](https://neurosys.com/blog/effective-coreference-resolution-model). | [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  <img width=250/>
