import os
import cv2
import torch
from  tqdm import tqdm
cateages = ['name','age','count','address','motivate']
D = [[{'text':'张三在网吧'},(0,1,'name'),(3,4,'address')],
     [{'text':'张三在美国'},(0,1,'name'),(3,4,'count')],
     [{'text':'张三在操场跑步'},(0,1,'name'),(3,4,'count'),(5,6,'motivate')],
     [{'text':'张三今年十八岁'},(0,1,'name'),(4,6,'age')],
     [{'text':'张三五十岁回中国'},(0,1,'name'),(2,4,'age'),(6,7,'count')],
     [{'text':'张三在打人'},(0,1,'name'),(3,4,'motivate')],
     ]
pred = [[(0,1,'name'),(3,4,'address')],
     [(0,1,'name'),(3,4,'count')],
     [(0,1,'name'),(3,4,'count'),(5,6,'motivate')],
     [(0,1,'name'),(4,6,'age')],
     [(0,1,'name'),(2,4,'age'),(6,7,'count')],
     [(0,1,'name'),(3,4,'motivate')],
     ]

def getsomecateage(data:list,cateage:str):
    newdata = set()
    for d in data:
        if d[-1] == cateage:
            newdata.add(d)
    return newdata

def evaluate(data,cateages,pred):
    res = {}
    for cateage in cateages:

        X,Y,Z = 0,0,0
        index = 0
        tmp = {}
        # for d in tqdm(data,ncols=60):
        for d  in data:
            R = set(pred[index])
            index += 1
            T = set([tuple(i) for i in d[1:]])
            R = getsomecateage(R,cateage)
            T = getsomecateage(T,cateage)
            X += len(R & T)
            Y += len(R)
            Z += len(T)
        Y = max(0,1e-10,Y)
        Z = max(0,1e-10,Z)
        f1,precision,recall = 2 * X/(Y+Z),X/Y,X/Z
        tmp['f1'] = f1
        tmp['precision'] = precision
        tmp['recall'] = recall
        res[cateage] = tmp
    return res
    # return f1,precision,recall

if __name__ == '__main__':
    res =  evaluate(D,cateages=['name'],pred=pred)
    for k ,v in res.items():
        print(k,v)


