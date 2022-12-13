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