'''
This uses BioSentVec for yes/no question answering.There is no training involved. 
The context and questions are converted into sentence embeddings by using BioSentVec and their similarity is measured.
In theory 'yes' questions should have more similarity with the context than the 'no' questions.
This code implements the above description.
'''
import argparse
from sklearn.metrics import accuracy_score
import os.path
from os import path
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
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


class DataHelper(Dataset):
	def __init__(self, data):
		self.data = torch.FloatTensor(data.astype('float'))
        
	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		target = self.data[index][-1]
		data_val = self.data[index] [:-1]
		return data_val,target


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

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
	def __init__(self,file_name,sim_thresh,sim_type,output_dir,model_path,epochs):
		self.file_name = file_name
		self.output_dir = output_dir
		self.model_path = model_path
		self.sim_type = sim_type
		self.model_path = "/mnt/nfs/work1/696ds-s20/kgunasekaran/sentvec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
		self.model = sent2vec.Sent2vecModel()
		self.batch_size =12
		self.learning_rate = 5e-4
		self.epochs =epochs	
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

	def train(self,model, device, train_loader, optimizer,criterion):
		model.train()
		y_true = []
		y_pred = []
		for i in train_loader:
			#LOADING THE DATA IN A BATCH
			data, target = i
		 
			#MOVING THE TENSORS TO THE CONFIGURED DEVICE
			data, target = data.to(device), target.to(device).long()
		       
			#FORWARD PASS
			output = model(data.float())
			#print(output.size()," ",target.size())
			loss = criterion(output, target) 
			
			#BACKWARD AND OPTIMIZE
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			# PREDICTIONS 
			pred = np.round(torch.argmax(output,dim=1).data.cpu().numpy())
			target = np.round(target.data.cpu().numpy())             
			y_pred.extend(pred.tolist())
			y_true.extend(target.tolist())
		print("Accuracy on training set is" ,accuracy_score(y_true,y_pred))

	
	#TESTING THE MODEL
	def eval(self,model, test_loader,criterion):
		#model in eval mode skips Dropout etc
		model.eval()
		y_true = []
		y_pred = []
		# set the requires_grad flag to false as we are in the test mode
		with torch.no_grad():
			for i in test_loader:
				#LOAD THE DATA IN A BATCH
				data,target = i
				# moving the tensors to the configured device
				data, target = data.to(device), target.to(device).long()
				# the model on the data
				output = model(data.float())
                        	# PREDICTIONS
				pred = np.round(torch.argmax(output,dim=1).data.cpu().numpy())
				target = np.round(target.data.cpu().numpy())
				y_pred.extend(pred.tolist())
				y_true.extend(target.tolist())
		print("Accuracy on test set is" , accuracy_score(y_true,y_pred))
		self.calculateMetrics(y_pred,y_true)

	def processTrain(self,train_dataset):
		#train dataset pandas variable
		
		train_helper = DataHelper(train_dataset)
		train_loader = torch.utils.data.DataLoader(dataset=train_helper, 
                                           batch_size=self.batch_size, 
                                           shuffle=True)


		self.input_size = 1400
		model = NeuralNet(self.input_size, 400, 2).to(device)
		#Loss and optimizer
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

		for i in range(0,self.epochs):
			print("Running epoch :",i)
			self.train(model, device, train_loader, optimizer,criterion)
			torch.save(model.state_dict(), self.output_dir+'epoch{}.pth'.format(i)) 
  
	def processEval(self,eval_dataset):
		#train dataset pandas variable
		eval_helper = DataHelper(eval_dataset)
		eval_loader = torch.utils.data.DataLoader(dataset=eval_helper,
                                           batch_size=self.batch_size,
                                           shuffle=True)
		self.input_size = 1400
		model = NeuralNet(self.input_size, 400, 2).to(device)
		#Loss and optimizer
		criterion = nn.CrossEntropyLoss()
		#optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
		if self.model_path!="":
                	model.load_state_dict(torch.load(self.model_path))       
		self.eval(model, eval_loader,criterion)



	def processQApairs(self,type,reload):
		qid_sim_hm = {}
		self.loadQuestionContextAnswer()
		sim_yes = []
		sim_no = []
		cnt_yes =0
		cnt_no = 0
		if type=="train" and reload==False:
			if path.exists("processed_biosentvec_data.npy"):
				self.processTrain(np.load("processed_biosentvec_data.npy"))
				return
		all_inps = np.zeros((1,1401), dtype=float)
		
		
		#print("ques vector:",ques_vec.shape)
		
		for i in range(0,len(self.bio_asq_data)):
			ques_vec = self.getEmbeddings(self.bio_asq_data[i].ques)
			context_vec = self.getEmbeddings(self.bio_asq_data[i].context)
			if i%100==0:
				print("i :",i)
			con_vec = np.concatenate((ques_vec,context_vec),axis=1) #,self.bio_asq_data[i].true_ans),axis=1)
			temp_ans = np.zeros((1,1),dtype=float)
			temp_ans[0,0] = self.bio_asq_data[i].true_ans 
			con_vec_merged = np.concatenate((con_vec,temp_ans),axis=1)
			all_inps = np.append(all_inps,con_vec_merged,axis = 0)
		if type=="train":
			np.save("processed_biosentvec_data.npy",all_inps)
			self.processTrain(all_inps)
		else:
			self.processEval(all_inps)
			
		

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
			#print("qid:",qid)
			self.qid_ans_hm[qid] = answer

			ba_data = BioASQ_data(question,context,answer,qid)
			self.bio_asq_data.append(ba_data)

#########################################################################################################################################################
				############################## Main file #######################################
#########################################################################################################################################################

parser = argparse.ArgumentParser(description='ADD YOUR DESCRIPTION HERE')
parser.add_argument('-type','--type', help='Train or Eval required (train/eval)',required=True)
parser.add_argument('-input','--input', help='Input file required',required=True)
parser.add_argument('-out','--output', help='Output folder required',required=True)
parser.add_argument('-force_convert','--reload', help='Convert to embeddings again',required=True)

parser.add_argument('-model_path','--model_path', help='set the model path to load it',required=False)
parser.add_argument('-epochs','--epochs', help='epochs',required=True)
args = parser.parse_args()


#n = len(sys.argv)
index=1
epochs =int(args.epochs)

# File path to dataset
file_name = args.input #"../../data/7b_train_files/BioASQ-train-yesno-7b-snippet.json
output_dir = args.output


# Threshold set to distinguish yes/no
sim_thresh = 0.5

sim_type = 1
#1 - cosine type, else - dot type 

print("Processing .......")
reload = False
if args.reload=="yes":
	reload = True

model_path =""
if args.model_path!=".":
	model_path = args.model_path

bv_yesno = BioSentVec_Yes_no(file_name,sim_thresh,sim_type,output_dir,model_path,epochs)

bv_yesno.processQApairs(args.type,reload)
