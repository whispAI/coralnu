## worklog for GPU refactor work
-- 2022-12-13 @lucafrost

**Environment Stuff**
* Install dependencies, namely `spacy==2.1` + `neuralcoref` + `allennlp`
* (Working in SageMaker) set instance type to [`ml.g4dn.xlarge`](https://instances.vantage.sh/aws/ec2/g4dn.xlarge) to get NVIDIA T4 GPU
* Having issues with `gcc` and other CLang stuff, also `jsonnet` is proving tricky, reverting to EC2 instance...
    * pushing only .ipynb changes to central repo, done...
    * spun up a `g4dn.xlarge` instance for access to 1x NVIDIA T4 GPU
        * running Deep Learning API GPU-Optimised PyTorch 1.12.1 on Ubuntu 20.04
	        * torch is accessible via `conda activate pytorch`
    * **neuralcoref:** uses PyTorch and numpy - requires torch >=1.3 and <1.4
        * whereas, **AllenNLP** requires torch < 1.13.0

**The Plan**
Duplicate PyTorch conda env & install AllenNLP spanbert-coref model, verify that operates correctly (quick lil python file)
Again, duplicate the base PyTorch env and install neuralcoref, validate install & functionality with a python file

Run pip freeze in both envs and check for dependency conflicts, HOPEFULLY we can get neuralcoref to execute within the AllenNLP dependencies, if not we may well have to dockerise

**Neuralcoref**
* Looks like the neuralcoref team *do not* wish to support GPU inference, nonetheless, it appears possible from the following PR [huggingface/neuralcoref/pull/149](https://github.com/huggingface/neuralcoref/pull/149)
    * This is going to be fun and games w/ dependencies, will spin up an EC2 instance later... attempting AllenNLP CUDA-isation for now.

**AllenNLP**
* clone "base" PyTorch environment with `conda create --name allennlp --clone pytorch`
* *install AllenNLP*
	* `‌pip install allennlp`
		* successfully, no CLang issues 
	* `pip install --pre allennlp-models`
* cool, let's make a python file to check AllenNLP Coreference w/ spanbert is working

checking CUDA is configured correctly, yup let's gooooo
```py
import torch
x = torch.randn(10, 10, device='cuda')
print(x)

>>> tensor([[ 1.0842, -1.5077,  0.7727, -0.7739, -0.3091, -1.0298,  0.5765,  1.4180,
         -0.4931, -0.0229],
        [ 2.0574,  0.6545, -1.7325, -2.0486, -0.0416, -0.4743,  0.4831, -0.1286,
         -1.0940, -0.0522],
        [ 1.2186,  0.5218,  0.2044, -1.5316,  0.9458,  0.8700, -0.5976,  1.3752,
         -1.3626,  1.4916],
        [-2.0960,  0.2086, -0.3420,  1.3386,  0.3196, -1.1668, -0.5033,  2.2678,
          0.8752, -1.1444],
        [-0.1125,  2.0883, -1.2472,  0.0556, -0.9477, -1.5637, -0.3471,  1.4461,
          1.0855, -1.4811],
        [-0.2838, -0.1701, -0.3638, -1.2668, -1.4179, -1.6126, -1.3225, -0.2605,
          1.2396, -1.7641],
        [ 0.9657, -0.1695, -0.9226, -1.5664,  0.2485, -0.9193,  0.5004, -0.5998,
          1.5205, -2.2022],
        [-1.0511, -0.3005,  0.1469, -0.4424,  0.2120, -1.1367, -1.4537,  1.3959,
         -0.4901,  0.6669],
        [-0.3813, -0.5952,  1.2886, -0.6312,  0.8543,  0.4421, -0.0175,  0.4950,
          1.4255,  0.7827],
        [-1.2314, -0.7776,  0.2215, -0.7553, -1.6839,  0.3571, -0.6372, -0.5896,
         -0.9887,  0.9065]], device='cuda:0')
```

time to check AllenNLP
```py
from allennlp.predictors.predictor import Predictor

model_path = 'coref-spanbert-large-2020.02.27.tar.gz'
predictor = Predictor.from_path(model_path, cuda_device=0)

text = 'Austin Jermaine Wiley (born January 8, 1999) is an American basketball player. He currently plays for the Auburn Tigers in the Southeastern Conference. Wiley attended Spain Park High School in Hoover, Alabama, where he averaged 27.1 points, 12.7 rebounds and 2.9 blocked shots as a junior in 2015-16, before moving to Florida, where he went to Calusa Preparatory School in Miami, Florida, while playing basketball at The Conrad Academy in Orlando.'
prediction = predictor.predict(document=text)['clusters']
print(prediction)

>>> [[[0, 2], [16, 16], [28, 28], [40, 40], [65, 65]], [[62, 62], [74, 74]]]
```

**moving on to `neuralcoref`**
* duplicating allenNLP conda env with `conda create --name neural --clone allennlp`
    * if executing both /gpu/allenNLP-test.py & /gpu/neuralcoref-test.py works within this new env (after installation of `neuralcoref` dependencies), then we can confirm compatibility & proceed with switching AllenNLP inference to CUDA immediately.
        * getting neuralcoref to work with CUDA is a little more tricky, but possible as per the following PR [huggingface/neuralcoref/pull/149](https://github.com/huggingface/neuralcoref/pull/149)
* *installation of neuralcoref*
* `conda activate neural`
```bash
pip install spacy==2.1
python -m spacy download en_core_web_sm
pip install neuralcoref --no-binary neuralcoref
```
* spacy install fails with `error: command '/usr/bin/gcc' failed with exit code 1`, time to install all the CLang stuff
    * sudo apt-get update && sudo apt-get upgrade
    * conda update andaconda
    * conda install -c conda-forge gxx
    * conda install -c conda-forge cxx-compiler
    * sudo apt-get install python3-dev
        * already available
* create test file for `neuralcoref` in `gpu/neuralcoref-test.py`
```py
import spacy
import neuralcoref
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

def neural(text):
    doc = nlp(text)
    out = {
        "resolved": doc._.coref_resolved,
        "clusters": doc._.coref_clusters,
        "token_data": [[token.text, token.pos_, token.tag_]  for token in doc]
    }
    return out

text = 'Austin Jermaine Wiley (born January 8, 1999) is an American basketball player. He currently plays for the Auburn Tigers in the Southeastern Conference. Wiley attended Spain Park High School in Hoover, Alabama, where he averaged 27.1 points, 12.7 rebounds and 2.9 blocked shots as a junior in 2015-16, before moving to Florida, where he went to Calusa Preparatory School in Miami, Florida, while playing basketball at The Conrad Academy in Orlando.'
res = neural(text)
print(res)

>>> {'resolved': 'Wiley is an American basketball player. Wiley currently plays for the Auburn Tigers in the Southeastern Conference. Wiley attended Spain Park High School in Hoover, Alabama, where Wiley averaged 27.1 points, 12.7 rebounds and 2.9 blocked shots as a junior in 2015-16, before moving to Florida, where Wiley went to Calusa Preparatory School in Miami, Florida, while playing basketball at The Conrad Academy in Orlando.', 'clusters': [Wiley: [Austin Jermaine Wiley (born January 8, 1999), He, Wiley, he, he], Florida: [Florida, Florida]], 'token_data': [['Austin', 'PROPN', 'NNP'], ['Jermaine', 'PROPN', 'NNP'], ['Wiley', 'PROPN', 'NNP'], ['(', 'PUNCT', '-LRB-'], ['born', 'VERB', 'VBN'], ['January', 'PROPN', 'NNP'], ['8', 'NUM', 'CD'], [',', 'PUNCT', ','], ['1999', 'NUM', 'CD'], [')', 'PUNCT', '-RRB-'], ['is', 'VERB', 'VBZ'], ['an', 'DET', 'DT'], ['American', 'ADJ', 'JJ'], ['basketball', 'NOUN', 'NN'], ['player', 'NOUN', 'NN'], ['.', 'PUNCT', '.'], ['He', 'PRON', 'PRP'], ['currently', 'ADV', 'RB'], ['plays', 'VERB', 'VBZ'], ['for', 'ADP', 'IN'], ['the', 'DET', 'DT'], ['Auburn', 'PROPN', 'NNP'], ['Tigers', 'PROPN', 'NNPS'], ['in', 'ADP', 'IN'], ['the', 'DET', 'DT'], ['Southeastern', 'PROPN', 'NNP'], ['Conference', 'PROPN', 'NNP'], ['.', 'PUNCT', '.'], ['Wiley', 'PROPN', 'NNP'], ['attended', 'VERB', 'VBD'], ['Spain', 'PROPN', 'NNP'], ['Park', 'PROPN', 'NNP'], ['High', 'PROPN', 'NNP'], ['School', 'PROPN', 'NNP'], ['in', 'ADP', 'IN'], ['Hoover', 'PROPN', 'NNP'], [',', 'PUNCT', ','], ['Alabama', 'PROPN', 'NNP'], [',', 'PUNCT', ','], ['where', 'ADV', 'WRB'], ['he', 'PRON', 'PRP'], ['averaged', 'VERB', 'VBD'], ['27.1', 'NUM', 'CD'], ['points', 'NOUN', 'NNS'], [',', 'PUNCT', ','], ['12.7', 'NUM', 'CD'], ['rebounds', 'NOUN', 'NNS'], ['and', 'CCONJ', 'CC'], ['2.9', 'NUM', 'CD'], ['blocked', 'VERB', 'VBD'], ['shots', 'NOUN', 'NNS'], ['as', 'ADP', 'IN'], ['a', 'DET', 'DT'], ['junior', 'NOUN', 'NN'], ['in', 'ADP', 'IN'], ['2015', 'NUM', 'CD'], ['-', 'SYM', 'SYM'], ['16', 'NUM', 'CD'], [',', 'PUNCT', ','], ['before', 'ADP', 'IN'], ['moving', 'VERB', 'VBG'], ['to', 'ADP', 'IN'], ['Florida', 'PROPN', 'NNP'], [',', 'PUNCT', ','], ['where', 'ADV', 'WRB'], ['he', 'PRON', 'PRP'], ['went', 'VERB', 'VBD'], ['to', 'ADP', 'IN'], ['Calusa', 'PROPN', 'NNP'], ['Preparatory', 'PROPN', 'NNP'], ['School', 'PROPN', 'NNP'], ['in', 'ADP', 'IN'], ['Miami', 'PROPN', 'NNP'], [',', 'PUNCT', ','], ['Florida', 'PROPN', 'NNP'], [',', 'PUNCT', ','], ['while', 'ADP', 'IN'], ['playing', 'VERB', 'VBG'], ['basketball', 'NOUN', 'NN'], ['at', 'ADP', 'IN'], ['The', 'DET', 'DT'], ['Conrad', 'PROPN', 'NNP'], ['Academy', 'PROPN', 'NNP'], ['in', 'ADP', 'IN'], ['Orlando', 'PROPN', 'NNP'], ['.', 'PUNCT', '.']]}
```
* as the `neural` env works to execute the neuralcoref test file, have also tested it with AllenNLP as `spacy==2.1.0` is supported by both.
  * can now proceed with starting server... allenNLP can be GPU-enabled, neuralcoref will have to wait for now
* finally, tweak neuralcoref installation to run on GPU
    * 