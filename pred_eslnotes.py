#!/bin/python
from sentence_transformers import SentenceTransformer, util

class PredictESLnotes:
    def _get_sentence_model(self):
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        return model

    def compute_similarity(self, s1, s2):
        #Sentences are encoded by calling model.encode()
        model = self._get_sentence_model()
        #Compute embedding for both lists
        embeddings1 = model.encode(s1, convert_to_tensor=True)
        embeddings2 = model.encode(s2, convert_to_tensor=True)

        #Compute cosine-similarits
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

        #Output the pairs with their score
        for i in range(len(s1)):
            print("{} \t\t {} \t\t Score: {:.4f}".format(s1[i], s2[i], cosine_scores[i][i]))


if __name__ == "__main__":
    pe = PredictESLnotes()
    # Two lists of sentences
    s1 = ['The cat sits outside',
                 'A man is playing guitar',
                 'The new movie is awesome']

    s2 = ['The dog plays in the garden',
                  'A woman watches TV',
                  'The new movie is so great']

    pe.compute_similarity(s1, s2)
