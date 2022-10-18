<!--- WHISP DEVELOPMENT LOGO ~ RESPONSIVE TO LIGHT/DARK MODE --->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://i.imgur.com/xLIjgR0.png" height="30" align="right">
  <img align="right" src="https://i.imgur.com/aDti3wF.png" height="30">
</picture>
<br><br>

# coralnu 🪸 ~ coref resolution w `neuralcoref` + `AllenNLP`
> 💡 this repository stores the code for a flask inference wrapper for **coralnu** — a coreference resolution implementation that combines neuralcoref and AllenNLP — alongside scripts to deploy the code for inference using WSGI/Gunicorn.

### about coralnu
**coralnu** [cor-al-noo] performs **CO**reference **R**esolution with **Al**lennlp and **N**e**u**ralcoref using ensemble methods to achieve a fuzzy intersection. to combine the clusters identified by both spaCy `neuralcoref` and AllenNLP's `coref-spanbert-large`, coralnu uses a method of intersection that favours AllenNLP (owing to high GAP performance) as the ground truth, and includes all spans that partially overlap in `neuralcoref` and `AllenNLP` clusters, but prioritises the shorter span. Find out more in NeuroSYS's [blog post](https://neurosys.com/blog/effective-coreference-resolution-model) or browse the code.

### quickstart: make requests to the hosted endpoint
**endpoint is currently offline**
```python
import requests
import json

url = "https://nlp.whisp.dev/coref"

payload = json.dumps({
  "text": [
    "Born and raised in London, Daniel Day-Lewis excelled on stage at the National Youth Theatre, before being accepted at the Bristol Old Vic Theatre School, which he attended for three years"
  ]
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
```

## References

| Implementation | Description | License |
|----------------|-------------|---------|
[Huggingface NeuralCoref 4.0](https://github.com/huggingface/neuralcoref) | a CR extension for spaCy, `NeuralCoref` resolves coreference clusters using neural networks. More information can be found on their [blog post](https://medium.com/huggingface/state-of-the-art-neural-coreference-resolution-for-chatbots-3302365dcf30) and the [demo](https://huggingface.co/coref/). | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  <img width=250/> 
[AllenNLP coreference resolution](https://github.com/allenai/allennlp-models) | open-source project part of [AI2](https://allenai.org/) institute which introduced [ELMo](https://allennlp.org/elmo). Their span-ranking coreference resolution model is also premised upon a neural model. More information is provided by their [demo](https://demo.allennlp.org/coreference-resolution). | [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  <img width=250/> 
[NeuroSYS-pl/coreference-resolution](https://github.com/NeuroSYS-pl/coreference-resolution). | NeuroSYS implementation of 4 intersection methods on AllenNLP and Neuralcoref CR clusters. More information can be found on their [blog post](https://neurosys.com/blog/effective-coreference-resolution-model). | [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  <img width=250/>
