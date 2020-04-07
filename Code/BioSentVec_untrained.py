'''
This uses BioSentVec for yes/no question answering.There is no training involved. 
The context and questions are converted into sentence embeddings by using BioSentVec and their similarity is measured.
In theory 'yes' questions should have more similarity with the context than the 'no' questions.
This code implements the above description.
'''

import sent2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
import sys 
import numpy as np 
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import json

class BioASQ_data:
	def __init__(self,ques,context,answer,qid):
		self.ques = ques
		self.context = context
		self.true_ans = answer
		self.qid = qid
	def storeSimCosine(self,val):
		self.cosine_sim = val
	def storeSimDot(self,val):
		self.dot_sim = val

class BioSentVec_Yes_no:
	def __init__(self,file_name,sim_thresh,sim_type):
		self.file_name = file_name
		self.sim_type = sim_type
		self.model_path = "/mnt/nfs/work1/696ds-s20/kgunasekaran/sentvec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
		self.model = sent2vec.Sent2vecModel()
		
		try:
    			self.model.load_model(self.model_path)
		except Exception as e:
    			print("EXCEPTION:",e)
	
		print('model successfully loaded')
		self.stop_words = set(stopwords.words('english'))
		self.bio_asq_data  = []


	def preprocess_sentence(self,text):
	    text = text.replace('/', ' / ')
	    text = text.replace('.-', ' .- ')
	    text = text.replace('.', ' . ')
	    text = text.replace('\'', ' \' ')
	    text = text.lower()

	    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in self.stop_words]

	    return ' '.join(tokens)

	def getEmbeddings(self,sentence):
		sentence = self.preprocess_sentence(sentence)
		sentence_vector = self.model.embed_sentence(sentence)
		#print(sentence_vector)
		return sentence_vector

	def cosine_sim(self,sen_vec1,sen_vec2):
		cosine_sim = 1 - distance.cosine(sen_vec1, sen_vec2)
		#print('cosine similarity:', cosine_sim)
		return cosine_sim

	def loadFromJson(self,data_file):
		with open(data_file, 'r') as f:
			datastore = json.load(f)
		return datastore

	def dot_sim(self,sen_vec1,sen_vec2):
		dot_sim = np.dot(sen_vec1,np.transpose( sen_vec2))
		return dot_sim

	def processQApairs(self):
		qid_sim_hm = {}
		self.loadQuestionContextAnswer()
		sim_yes = []
		sim_no = []
		cnt_yes =0
		cnt_no = 0
		for i in range(0,len(self.bio_asq_data)):
			ques_vec = self.getEmbeddings(self.bio_asq_data[i].ques)
			context_vec = self.getEmbeddings(self.bio_asq_data[i].context)
			
			cos_sim = self.cosine_sim(ques_vec,context_vec)
			dot_sim =  self.dot_sim(ques_vec,context_vec)
			self.bio_asq_data[i].storeSimCosine(cos_sim)
			self.bio_asq_data[i].storeSimDot(dot_sim)
			if self.sim_type ==1:
				sim = cos_sim
			else:
				sim = dot_sim
			if self.bio_asq_data[i].qid in qid_sim_hm:
				qid_sim_hm[self.bio_asq_data[i].qid] = max(qid_sim_hm[self.bio_asq_data[i].qid],sim)
			else:
				qid_sim_hm[self.bio_asq_data[i].qid] = sim
			if self.bio_asq_data[i].true_ans==0:
				cnt_no=cnt_no+1
				sim_no.append(sim)
			else:
				cnt_yes = cnt_yes+1
				sim_yes.append(sim)
		np.save("sim_no.npy",np.asarray(sim_no))
		np.save("sim_yes.npy",np.asarray(sim_yes))
		preds,gt = self.createList(qid_sim_hm)
		self.calculateMetrics(preds,gt)

	def createList(self,qid_sim_hm):
		pred_list =[]
		gt_list = []
		for key in qid_sim_hm:
			pred_ans = 0 
			if qid_sim_hm[key]>0.5:
				pred_ans = 1
			pred_list.append(pred_ans)
			gt_list.append(self.qid_ans_hm[key])
		return pred_list,gt_list

	def calculateMetrics(self,preds,gt):
		accuracy = (100*np.sum(preds==gt))/len(preds)
		macro_f1 = f1_score(gt,preds, average='macro')
		target_names = ['No', 'Yes']
		print("preds size:",len(preds)," gt size:",len(gt))
		print("Accuracy:",accuracy," macro f1:",macro_f1)
		print("------------------------ ",classification_report(gt, preds, target_names=target_names),"------------------------------")
	
	def loadQuestionContextAnswer(self):
		data_json = self.loadFromJson(self.file_name)
		qa_pairs = data_json["data"][0]["paragraphs"]
		self.qid_ans_hm = {}
		print("Total question pairs:",len(qa_pairs))
		for i in range(0,len(qa_pairs)):
			context = self.preprocess_sentence(qa_pairs[i]["context"])
			question = self.preprocess_sentence(qa_pairs[i]["qas"][0]["question"])
			answer = 0
			if qa_pairs[i]["qas"][0]["answers"]=="yes":
				answer = 1
			qid = qa_pairs[i]["qas"][0]["id"]
			if len(qid)>24:
				qid = qid.split("_")[0]
			print("qid:",qid)
			self.qid_ans_hm[qid] = answer

			ba_data = BioASQ_data(question,context,answer,qid)
			self.bio_asq_data.append(ba_data)

#########################################################################################################################################################
				############################## Main file #######################################
#########################################################################################################################################################

n = len(sys.argv)
index=1

# File path to dataset
file_name = sys.argv[1] #"../../data/7b_train_files/BioASQ-train-yesno-7b-snippet.json" 

# Threshold set to distinguish yes/no
sim_thresh = 0.5

if n>index+1:
	sim_thresh = int(sys.argv[index+1]) 

sim_type = 1

if n>index+2:
	sim_type = int(sys.argv[index+2])

#1 - cosine type, else - dot type 

print("Processing .......")

bv_yesno = BioSentVec_Yes_no(file_name,sim_thresh,sim_type)
bv_yesno.processQApairs()
