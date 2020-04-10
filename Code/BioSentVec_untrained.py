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
	def __init__(self,file_name,sim_thresh,sim_type):
		self.file_name = file_name
		self.sim_type = sim_type
		#self.model_path = "/mnt/nfs/work1/696ds-s20/kgunasekaran/sentvec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
		self.model = sent2vec.Sent2vecModel()
		self.batch_size =12
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

	def train(self,model, device, train_loader, optimizer):
		model.train()
		y_true = []
		y_pred = []
		for i in train_loader:
			#LOADING THE DATA IN A BATCH
			data, target = i
		 
			#MOVING THE TENSORS TO THE CONFIGURED DEVICE
			data, target = data.to(device), target.to(device)
		       
			#FORWARD PASS
			output = model(data.float())
			loss = criterion(output, target.unsqueeze(1)) 
			
			#BACKWARD AND OPTIMIZE
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			# PREDICTIONS 
			pred = np.round(output.detach())
			target = np.round(target.detach())             
			y_pred.extend(pred.tolist())
			y_true.extend(target.tolist())
		print("Accuracy on training set is" ,accuracy_score(y_true,y_pred))

	'''
	#TESTING THE MODEL
	def test(self,model, device, test_loader):
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
		    data, target = data.to(device), target.to(device)
		    
		    # the model on the data
		    output = model(data.float())
		               
		    #PREDICTIONS
		    pred = np.round(output)
		    target = target.float()
		    y_true.extend(target.tolist()) 
		    y_pred.extend(pred.reshape(-1).tolist())
	    print("Accuracy on test set is" , accuracy_score(y_true,y_pred))
	'''
	def processTrain(self,train_dataset):
		#train dataset pandas variable
		
		train_helper = DataHelper(train_dataset)
		train_loader = torch.utils.data.DataLoader(dataset=train_helper, 
                                           batch_size=self.batch_size, 
                                           shuffle=True)


		self.input_size = 700
		model = NeuralNet(self.input_size, 200, 2).to(device)
		# Loss and optimizer
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

		for i in range(0,self.epochs):
			self.train(model, device, train_loader, optimizer)
			#torch.save(model.state_dict(), 'train_valid_exp4-epoch{}.pth'.format(epoch)) 
  
	def processQApairs(self):
		qid_sim_hm = {}
		self.loadQuestionContextAnswer()
		sim_yes = []
		sim_no = []
		cnt_yes =0
		cnt_no = 0
		'''
		ques_vec = self.getEmbeddings(self.bio_asq_data[:].ques)
		context_vec = self.getEmbeddings(self.bio_asq_data[:].context)
		con_vec = np.concatenate(np.concatenate((ques_vec,context_vec),axis=1),self.bio_asq_data[:].true_ans)
		'''

		all_inps = np.zeros((1,1401), dtype=float)

		
		
		#print("ques vector:",ques_vec.shape)
		
		for i in range(0,len(self.bio_asq_data)):
			ques_vec = self.getEmbeddings(self.bio_asq_data[i].ques)
			context_vec = self.getEmbeddings(self.bio_asq_data[i].context)
			
			con_vec = np.concatenate((np.concatenate((ques_vec,context_vec),axis=1),self.bio_asq_data[i].true_ans),axis=1)
			all_inps = np.append(all_inps,con_vec,axis = 0)

		self.processTrain(all_inps)
			
		

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
