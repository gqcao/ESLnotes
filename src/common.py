def loadstr(self, filename, converter=str):
    return [converter(c.strip()) for c in open(filename).readlines()]

def writestr(self, filename, texts):
    with open(filename, 'w') as outfile:
        for i in range(len(texts)):
            line = str(texts[i]) + '\n'
            outfile.write(line)

def save2svm(self, filename, data, row_names):
    with open(filename, 'w') as outfile:
        for cnt,x in enumerate(data):
            x = array(x)
            indexes = x.nonzero()[0]
            values = x[indexes]
            pairs = ['%i:%f'%(indexes[i]+1,values[i]) for i in range(len(indexes))]
            sep_line = ['+1']
            sep_line.extend(pairs)
            sep_line.append('#'+ row_names[cnt])
            sep_line.append('\n')
            line = ' '.join(sep_line)
            outfile.write(line)
