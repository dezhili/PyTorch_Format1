import os
from torch.utils.data import Dataset
import random
from PIL import Image

'''
Data storage format: class mixed

../data/train/xxx.jpg  
../data/test/xxx.jpg

'''


class DogCat(data.Dataset):
	def __init__(self,root,transform=None,train=True,val=False,test=False):
		self.test=test
		self.train=train
		self.val=val
		self.transform=transform
		imgs=[os.path.join(root,img)for img in os.listdir(root)]

		# test: data/test/8973.jpg
		# train: data/train/cat.10004.jpg

		if self.test:
			imgs=sorted(imgs,key=lambda x: int(x.split('.')[-2].split('/')[-1]))
		else:
			imgs=sorted(imgs,key=lambda x: int(x.split('.')[-2]))

		imgs_num=len(imgs)
		
		if self.test:
			self.imgs=imgs
		else:
			random.shuffle(imgs)
			if self.train:
				self.imgs=imgs[:int(0.7*imgs_num)]
			elif self.val:
				self.imgs=imgs[int(0.7*imgs_num):]
				
	def __getitem__(self,index):
		img_path=self.imgs[index]
		if self.test:
			label=int(self.imgs[index].split('.')[-2].split('/')[-1])
		else:
			label=1 if 'dog' in img_path.split('/')[-1] else 0
		data=Image.open(img_path)
		data=self.transform(data)
		return data,label
	def __len__(self):
		return len(self.imgs)