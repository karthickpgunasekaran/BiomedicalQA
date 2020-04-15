'''
This uses BioSentVec for yes/no question answering. 
The context and questions are converted into sentence embeddings by using BioSentVec.
The sentence and context embeddings are concatinated and feed into two fully concatenated layers followed by softmax to produce yes/no class probs.
Cross entropy loss and adam optimizer is used for updating the weights in back prop.
This following code implements the above description.
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
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

# This the pytorch custom datatuple which will be used by pytorch dataloader 
class DataHelper(Dataset):
	def __init__(self,qid,data):
		self.data = torch.FloatTensor(data.astype('float'))
		#if qid==None:
		#self.qid = "00"
		#else:
		self.qid = qid
	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		target = self.data[index][-1]
		data_val = self.data[index] [:-1]
		if self.qid==None:
			return data_val,target,"qwerty"
		#print("index:",index)
		return data_val,target,self.qid[index]


# Fully connected neural network with two layer. BioSentVec embeddings will be concatenated and fed into it. Output will be either yes/no here.
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout = nn.Dropout(0.10)
        #self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc1_1 = nn.Linear(input_size,2) 
        self.relu = nn.ReLU()
        #self.fc2 = nn.Linear(hidden_size, num_classes)
        self.fc1_5 = nn.Linear(1400,1100)
        self.fc2_5 = nn.Linear(1100,800)
        self.fc3_5 = nn.Linear(800,500)
        self.fc4_5 = nn.Linear(500,250)
        self.fc5_5 = nn.Linear(250,2)
        self.fcnew = nn.Linear(input_size,num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        out = self.dropout(x)
        #one layer
        #out = self.fc1_1(out)
        #two layer
        
        '''
        out = self.fc1_5(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2_5(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3_5(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc4_5(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc5_5(out)
        #out = self.fc3(out)
        out = self.softmax(out)
        '''
        return out


class NeuralNet_2layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet_2layer, self).__init__()
        self.dropout = nn.Dropout(0.10)
        self.fc1 = nn.Linear(input_size, 700)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(700,num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.dropout(x)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out


class NeuralNet_3layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet_3layer, self).__init__()
        self.dropout = nn.Dropout(0.10)
        self.fc1 = nn.Linear(input_size, 800)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(800,400)
        self.fc3 = nn.Linear(400,num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.dropout(x)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out

# BioASQ json data file is stored as a dataset here
class BioASQ_data:
	def __init__(self,ques,context,answer,qid):
		self.ques = ques
		self.context = context
		self.true_ans = answer
		self.qid = qid

#This class processes and controls all the training and evaluation using the biosentvec.
class BioSentVec_Yes_no:
	#init function to intialize all the variables necessary
	def __init__(self,train_file,eval_file,output_dir,model_path,force_create_emp,run_type,epochs):
		self.train_file = train_file
		self.eval_file = eval_file
		self.output_dir = output_dir
		self.model_path = model_path
		self.force_create_emb = force_create_emp
		self.biosentvec_model_path = "/mnt/nfs/work1/696ds-s20/kgunasekaran/sentvec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
		self.biosentvec_model = sent2vec.Sent2vecModel()
		# hyparameters to be used and played around with
		self.batch_size = 32
		self.run_type = run_type
		self.learning_rate = 11e-6 #3- 9e-6
		self.input_size = 1400
		self.hidden_size = 700
		self.output_size = 2
		self.unbalanced =9200
		self.weight_decay = 1e-3 #1e-1
		self.epochs = 16 #epochs
		self.no_of_layers =2
		#end hyperparameters
		try:
    			print("loading biosentvec model..")
    			self.biosentvec_model.load_model(self.biosentvec_model_path)
		except Exception as e:
    			print("EXCEPTION:",e)
	
		print('model successfully loaded')
		self.stop_words = set(stopwords.words('english'))
		self.bio_asq_data_train  = []
		self.bio_asq_data_eval = []
		self.qid_pred = {}
		self.qid_pred_prob = {}
		self.qid_target = {}

	#default preprocessing suggested in BioSentVec documentation.
	def preprocess_sentence(self,text):
	    text = text.replace('/', ' / ')
	    text = text.replace('.-', ' .- ')
	    text = text.replace('.', ' . ')
	    text = text.replace('\'', ' \' ')
	    text = text.lower()
	    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in self.stop_words]
	    return ' '.join(tokens)

	#This function returns BioSentVec embeddings for a sentence.
	def getEmbeddings(self,sentence):
		sentence = self.preprocess_sentence(sentence)
		sentence_vector = self.biosentvec_model.embed_sentence(sentence)
		#print(sentence_vector)
		return sentence_vector

	
	#This function loads json data
	def loadFromJson(self,data_file):
		with open(data_file, 'r') as f:
			datastore = json.load(f)
		return datastore

	#This function helps to finetuning layers with embeddings from biosentvec. This function is called for each epoch.
	def train(self):
		self.model.train()
		y_true = []
		y_pred = []
		for i in self.train_loader:
			#LOADING THE DATA IN A BATCH
			data, target,_ = i
		 
			#MOVING THE TENSORS TO THE CONFIGURED DEVICE
			data, target = data.to(device), target.to(device).long()
		       
			#FORWARD PASS
			output = self.model(data.float())
			#print("data : ",data.float())
			#print(output.size()," ",target.size())
			#print("target:",target)
			#print("output:",output)
			loss = self.criterion(output, target) 
			#print("loss: ",loss.data)
			#BACKWARD AND OPTIMIZE
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			
			# PREDICTIONS 
			pred = np.round(torch.argmax(output,dim=1).data.cpu().numpy())
			target = np.round(target.data.cpu().numpy())             
			y_pred.extend(pred.tolist())
			y_true.extend(target.tolist())
		print("Accuracy on training set is" ,accuracy_score(y_true,y_pred))

	#This function stores the predictions and target values for each qid. Highest probability prediction from softmax is stored here for each qid. 
	def savePredTrue(self,qid,pred,pred_prob,target):
		for i in range(0,len(pred)):
			#print("Qid:",qid[i])
			self.qid_target[qid[i]] = target[i]
			if qid[i] in self.qid_pred:
				#print("if part ",self.qid_pred_prob[qid[i]])
				if self.qid_pred_prob[qid[i]] < pred_prob[i]:
					self.qid_pred_prob[qid[i]] = pred_prob[i]
					self.qid_pred[qid[i]] = pred[i]
			else:
				#print("else part")
				self.qid_pred_prob[qid[i]] = pred_prob[i]
				self.qid_pred[qid[i]] = pred[i]
			#print("dict len:",len(self.qid_pred))
	#This function creates list of preds and true values from the dictionary created in savePredTrue().
	def createListForAnswers(self):
		pred = []
		true =[]
		print("No of elements in dict: ",len(self.qid_pred))
		for i in self.qid_pred:
			pred.append(self.qid_pred[i])
			true.append(self.qid_target[i])
		return pred,true

	#Evaluation of model is carried out in this function.
	def eval(self):
		#model in eval mode skips Dropout etc
		self.model.eval()
		y_true = []
		y_pred = []
		# set the requires_grad flag to false as we are in the test mode
		with torch.no_grad():
			for i in self.eval_loader:
				#LOAD THE DATA IN A BATCH
				data,target,qid = i
				# moving the tensors to the configured device
				data, target = data.to(device), target.to(device).long()
				# the model on the data
				output = self.model(data.float())
                        	# PREDICTIONS
				pred = np.round(torch.argmax(output,dim=1).data.cpu().numpy())
				pred_prob = torch.max(output,dim=1)[0].data.cpu().numpy()
				#print("pred prob:",pred_prob)
				target = np.round(target.data.cpu().numpy())
				self.savePredTrue(qid,pred,pred_prob,target)
				#y_pred.extend(pred.tolist())
				#y_true.extend(target.tolist())
		#print("Accuracy on test set is" , accuracy_score(gt,preds))
		preds, gt = self.createListForAnswers()
		print("Accuracy on test set is" , accuracy_score(gt,preds))
		print("Final predictions count:",len(gt))
		self.calculateMetrics(preds,gt)

	#This function carries out training depending on no of epochs.
	def processTrain(self):
		#train dataset pandas variable
		for i in range(0,self.epochs):
			print("Running epoch :",i)
			self.train()
			torch.save(self.model.state_dict(), self.output_dir+'epoch{}.pth'.format(i))
 
	#This function helps carry out evaluation.
	def processEval(self):
		#train dataset pandas variable
		self.eval()

	#This function helps with both training and evaluating together after each epoch
	def processTrainEval(self):
		#train dataset pandas variable
		for i in range(0,self.epochs):
			self.qid_pred = {}
			self.qid_pred_prob = {}
			self.qid_target = {}
			print("Running epoch :",i)
			self.train()
			torch.save(self.model.state_dict(), self.output_dir+'epoch{}.pth'.format(i)) 
			self.eval()

  	#This function initializes all the datalaoders and models required for training and testing
	def createDataLoaderAndModels(self):
		#load and init model
		if self.no_of_layers==2:
			self.model = NeuralNet_2layer(self.input_size, self.hidden_size, self.output_size).to(device)
		else:
			self.model = NeuralNet_3layer(self.input_size, self.hidden_size, self.output_size).to(device)

		if self.model_path!="-":
			print("loading pretrained model...")
			self.model.load_state_dict(torch.load(self.model_path))
		self.criterion = nn.CrossEntropyLoss()
		
		#initialize for training
		if self.run_type==2 or self.run_type==0:
			train_helper = DataHelper(None,self.train_emb_data)
			self.train_loader = torch.utils.data.DataLoader(dataset=train_helper, 
                                           batch_size=self.batch_size, 
                                           shuffle=True)
			
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay)
		#initialize for evaluation
		if self.run_type==2 or self.run_type==1:
			eval_helper = DataHelper(self.eval_qid,self.eval_emb_data)
			self.eval_loader = torch.utils.data.DataLoader(dataset=eval_helper,
                                           batch_size=self.batch_size)

	# This function converts the entire dataset to BioSentVec embeddings
	def convertToEmbeddings(self,inp_data):
		qid = []
		embs = np.zeros((1,1401), dtype=float)
		for i in range(0,len(inp_data)):
			ques_vec = self.getEmbeddings(inp_data[i].ques)
			context_vec = self.getEmbeddings(inp_data[i].context)
			if i%100==0:
				print("i :",i)
			qid.append(inp_data[i].qid)			
			con_vec = np.concatenate((ques_vec,context_vec),axis=1) 
			temp_ans = np.zeros((1,1),dtype=float)
			temp_ans[0,0] = inp_data[i].true_ans 
			con_vec_merged = np.concatenate((con_vec,temp_ans),axis=1)
			embs = np.append(embs,con_vec_merged,axis = 0)
		return embs[1:][:],qid

	# This function helps load all the question context pairs from the json file
	def loadQuestionContextAnswer(self,file_name,run_type):
		data_json = self.loadFromJson(file_name)
		qa_pairs = data_json["data"][0]["paragraphs"]
		self.qid_ans_hm = {}
		bio_asq_data_yes =[]
		bio_asq_data_no =[]

		print("Total question pairs:",len(qa_pairs))
		for i in range(0,len(qa_pairs)):
			context = self.preprocess_sentence(qa_pairs[i]["context"])
			question = self.preprocess_sentence(qa_pairs[i]["qas"][0]["question"])
			answer = 0
			if qa_pairs[i]["qas"][0]["answers"]=="yes":
				answer = 1
			qid = qa_pairs[i]["qas"][0]["id"]
			if "_" in qid:#len(qid)>24:
				qid = qid.split("_")[0]
			#print("qid:",qid)
			self.qid_ans_hm[qid] = answer

			ba_data = BioASQ_data(question,context,answer,qid)
			if run_type==0:
				if answer==1:
					bio_asq_data_yes.append(ba_data)
				else:
					bio_asq_data_no.append(ba_data)
			else:
				bio_asq_data_yes.append(ba_data)
		return bio_asq_data_yes,bio_asq_data_no

	def mainController(self):
		
		if self.run_type==0 or self.run_type==2:
			if not (self.force_create_emb==False and path.exists("processed_biosentvec_emb_train.npy")):
				bio_yes, bio_no = self.loadQuestionContextAnswer(self.train_file,0)
				print("train yes ques:",len(bio_yes)," train no ques: ",len(bio_no))
				self.bio_asq_data_train = bio_yes[:self.unbalanced] + bio_no[:self.unbalanced]#bio_yes+bio_no+bio_no+bio_no+bio_no+bio_no+bio_no+bio_no #bio_yes[:self.unbalanced] + bio_no[:self.unbalanced]
				random.shuffle(self.bio_asq_data_train)
				self.train_emb_data,_ = self.convertToEmbeddings(self.bio_asq_data_train)
				np.save("processed_biosentvec_emb_train.npy",self.train_emb_data)
			self.train_emb_data = np.load("processed_biosentvec_emb_train.npy")

		if self.run_type==1 or self.run_type==2:
			self.bio_asq_data_eval, _ = self.loadQuestionContextAnswer(self.eval_file,1)
			#print("eval yes ques:",len(bio_yes)," eval no ques: ",len(bio_no))
			self.eval_emb_data,self.eval_qid = self.convertToEmbeddings(self.bio_asq_data_eval)

		self.createDataLoaderAndModels()

		if self.run_type == 2:
			self.processTrainEval()
		if self.run_type == 0:
			self.processTrain()
		if self.run_type == 1:
			self.processEval()

	def calculateMetrics(self,preds,gt):
		accuracy = (100*np.sum(preds==gt))/len(preds)
		macro_f1 = f1_score(gt,preds, average='macro')
		target_names = ['No', 'Yes']
		print("preds size:",len(preds)," gt size:",len(gt))
		print("Accuracy:",accuracy," macro f1:",macro_f1)
		print("------------------------ ",classification_report(gt, preds, target_names=target_names),"------------------------------")
	


#########################################################################################################################################################
				############################## Main file #######################################
#########################################################################################################################################################

parser = argparse.ArgumentParser(description='python BioSentVec_yesno [-train_input training_file_name] [-eval_input evaluation_file_name] [-out_dir output_folder_for_saving_model] [-force_convert (yes/no) - recreates embeddings again even if availble] [-model_path load_pretrained_model] [-epochs specify_no_of_epochs] ')

parser.add_argument('-train_input','--train_input', help='Input file required',default="-",required=False)

parser.add_argument('-eval_input','--eval_input', help='Eval file required',default="-",required=False)

parser.add_argument('-out_dir','--output', help='Output folder required',default="temp_biosentvec_folder/",required=False)

parser.add_argument('-force_convert','--reload', help='Convert to embeddings again',default="no",required=False)

parser.add_argument('-model_path','--model_path', help='set the model path to load it',default="-",required=False)

parser.add_argument('-epochs','--epochs', help='epochs',default="5",required=False)

args = parser.parse_args()





# File path to dataset
train_file = args.train_input 
output_dir = args.output
eval_file = args.eval_input

#run type 0-train, 1- eval, 2 -both 
if eval_file=="-" and train_file=="-":
	print("Neither eval file nor train file is specified")
	raise ValueError('Neither eval file nor train file is specified as arguments')
elif eval_file=="-":
	run_type = 0
elif train_file=="-":
	run_type = 1
else:
	run_type = 2


print("Processing .......")
reload_file = False
if args.reload=="yes":
	reload_file = True

model_path = args.model_path

epochs =int(args.epochs)

bv_yesno = BioSentVec_Yes_no(train_file,eval_file,output_dir,model_path,reload_file,run_type,epochs)
print("Starting controller")
bv_yesno.mainController()
