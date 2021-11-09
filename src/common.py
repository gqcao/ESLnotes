import re
import numpy as np

def loadstr(filename, converter=str):
    return [converter(c.strip()) for c in open(filename).readlines()]

def writestr(filename, texts):
    with open(filename, 'w') as outfile:
        for i in range(len(texts)):
            line = str(texts[i]) + '\n'
            outfile.write(line)

def save2svm(filename, data, row_names):
    with open(filename, 'w') as outfile:
        for cnt,x in enumerate(data):
            x = np.array(x)
            indexes = x.nonzero()[0]
            values = x[indexes]
            pairs = ['%i:%f'%(indexes[i]+1,values[i]) for i in range(len(indexes))]
            sep_line = ['+1']
            sep_line.extend(pairs)
            sep_line.append('#'+ row_names[cnt])
            sep_line.append('\n')
            line = ' '.join(sep_line)
            outfile.write(line)

def get_sentence_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model
