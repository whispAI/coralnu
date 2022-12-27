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
  "clusters" : [[[6,9],[31,31]]],
  "method" : "fuzzyIntersection",
  "resolved" : "Born and raised in London, Daniel Day-Lewis excelled on stage at the National Youth Theatre, before being accepted at the Bristol Old Vic Theatre School, which Daniel Day-Lewis attended for 3 years"
}
```

### deployment guide
**N.B.** these instructions have been tested on EC2 `g4dn` instances, equipped with NVIDIA T4 GPUs, running Amazon Linux 2. Some modifications may be necessary to deploy Coralnu on Ubuntu and other operating systems. This guide assumes a 'clean slate' on the server prior to installation.

1. Install `mamba` package manager: `cd /tmp/ && wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh`
  * Execute the shell file you just downloaded, with `bash Mambaforge-Linux-x86_64.sh` and follow the on-screen prompts. Ensure to respond with 'yes' when prompted whether to run `conda init`.
  * Once Mamba has been installed, terminate the ssh connection and re-launch your shell. Upon re-connecting to the instance, you should be in the base environment.
2. Create a new environment for Coralnu with `mamba create -n coralnu python=3.7` and `mamba activate coralnu`
3. Install CLang dependencies needed for CUDA, AllenNLP, and Neuralcoref...
  * `mamba install -c conda-forge gcc cxx-compiler` and `sudo yum install -y gcc kernel-devel-$(uname -r)`
  * add CC (gcc) to your path with `sudo CC=/usr/bin/gcc10-cc`
4. You will now need to install the NVIDIA/CUDA drivers from AWS S3, this guide assumes that you have `awscli` configured.
  * Download the drivers with `aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ .` then execute the driver `./NVIDIA-Linux-x86_64*.run`
    * For further information, see [AWS Docs: Install NVIDIA drivers on Linux instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html#nvidia-driver-instance-type)
  * Check whether the installation was successful with `nvidia-smi -q | head`
5. Install Coralnu dependencies
  * `pip install spacy=2.1.0 allennlp neuralcoref Flask`
  * `pip install --pre allennlp-models`
  * `python -m spacy download en_core_web_sm`
6. Install `git` and clone the Coralnu repo
  * `mamba install git && cd ~/ && git clone https://github.com/whispAI/coralnu.git`
  * Download the spanBERT model with `cd ~/coralnu/gpu && wget https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz`
7. Run the server with `python app.py`

```sh
#                    _   ʕっ•ᴥ•ʔっ       
#    __ ___ _ _ __ _| |_ _ _  _ 
#   / _/ _ \ '_/ _` | | ' \ || |
#   \__\___/_| \__,_|_|_||_\_,_|
#                             
# INSTALLATION WALKTHROUGH
# ~ working on Amazon Linux 2 on EC2 g4dn.xlarge instance ~

sudo yum update
sudo yum upgrade
cd /tmp/
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh
# follow mamba setup procedure
# close terminal/ssh connection & reconnect

# install clang deps
mamba create -n coralnu python=3.7
mamba activate coralnu
mamba install git
mamba install -c conda-forge gcc
mamba install -c conda-forge cxx-compiler
sudo yum install -y gcc kernel-devel-$(uname -r)

# install NVIDIA drivers ~ N.B. IGNORE `cc` version check
aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ .
chmod +x NVIDIA-Linux-x86_64*.run
sudo CC=/usr/bin/gcc10-cc ./NVIDIA-Linux-x86_64*.run
# reboot the instance & confirm CUDA is functional
# option 1
nvidia-smi -q | head
# option 2
python ~/coralnu/gpu/gpu-test.py

# install coralnu deps
pip install spacy=2.1.0 allennlp neuralcoref Flask
pip install --pre allennlp-models
python -m spacy download en_core_web_sm
git clone https://github.com/whispAI/coralnu.git
cd coralnu/gpu/
# only required if model hasn't been downloaded via 
git-lfs
wget https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz

# run the server (in debug mode)
python app.py

```

## References

| Implementation | Description | License |
|----------------|-------------|---------|
[Huggingface NeuralCoref 4.0](https://github.com/huggingface/neuralcoref) | a CR extension for spaCy, `NeuralCoref` resolves coreference clusters using neural networks. More information can be found on their [blog post](https://medium.com/huggingface/state-of-the-art-neural-coreference-resolution-for-chatbots-3302365dcf30) and the [demo](https://huggingface.co/coref/). | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  <img width=250/> 
[AllenNLP coreference resolution](https://github.com/allenai/allennlp-models) | open-source project part of [AI2](https://allenai.org/) institute which introduced [ELMo](https://allennlp.org/elmo). Their span-ranking coreference resolution model is also premised upon a neural model. More information is provided by their [demo](https://demo.allennlp.org/coreference-resolution). | [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  <img width=250/> 
[NeuroSYS-pl/coreference-resolution](https://github.com/NeuroSYS-pl/coreference-resolution). | NeuroSYS implementation of 4 intersection methods on AllenNLP and Neuralcoref CR clusters. More information can be found on their [blog post](https://neurosys.com/blog/effective-coreference-resolution-model). | [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  <img width=250/>
