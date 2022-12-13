import allennlp
from allennlp.predictors.predictor import Predictor

model_path = 'coref-spanbert-large-2020.02.27.tar.gz'
predictor = Predictor.from_path(model_path, cuda_device=0)

text = 'Austin Jermaine Wiley (born January 8, 1999) is an American basketball player. He currently plays for the Auburn Tigers in the Southeastern Conference. Wiley attended Spain Park High School in Hoover, Alabama, where he averaged 27.1 points, 12.7 rebounds and 2.9 blocked shots as a junior in 2015-16, before moving to Florida, where he went to Calusa Preparatory School in Miami, Florida, while playing basketball at The Conrad Academy in Orlando.'
prediction = predictor.predict(document=text)['clusters']
print(prediction)