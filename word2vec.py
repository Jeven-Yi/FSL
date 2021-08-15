import torch
import torch.nn as nn
import jieba
from gensim.models import Word2Vec as wtv
raw_text = [
    "你站在桥上看风景",
	"看风景的人在楼上默默地看着你",
	"明月装饰了你的窗子",
	"你装饰了别人的梦",
	"我觉得你说的很对"
]
texts = [[word for word in jieba.cut(text,cut_all=True)] for text in raw_text]
print(texts)
print(len(texts))
model = wtv(texts,min_count=2,window=5,sg=1,size=10,iter=100)
print('finish')
model.save(r'D:\nlp\wordvec' + '/' + 'wzh_word2vec.model')
model.wv.save_word2vec_format(r'D:\nlp\wordvec' + '/' + 'word.txt',binary=False)
pre_model = wtv.load(r'D:\nlp\wordvec' + '/' + 'wzh_word2vec.model')
words = list(pre_model.wv.vocab)
vocab_size = len(words)
word_vectors = torch.randn([vocab_size,10])
for i in range(vocab_size):
	vector = pre_model[words[i]]
	word_vectors[i,:] = torch.from_numpy(vector)

embedding = nn.Embedding.from_pretrained(word_vectors)
embedding.weight.requires_grad = False
for k,v in embedding.state_dict().items():
	print(k,v)
