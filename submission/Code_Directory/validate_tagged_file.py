# validated that the competition file without the tags is the same as the original file

with open('comp.tagged') as f:
    sentences = f.readlines()

# get a list of sentences and a list of sentences tags
words = [[tuple.split('_')[0] for tuple in sentence.strip().split(' ')] for sentence in sentences]

for sentence in words:
    print(' '.join(sentence))
