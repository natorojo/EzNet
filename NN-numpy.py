import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import datasets

def normalize(X,ax):
	#center
	X = X - np.mean(X,axis=ax).reshape(-1,1)
	#subtract std
	X = X/np.std(X,axis=ax).reshape(-1,1)
	return X

class FFNN:
	def __init__(self,opts):
		"""
			A simple FFNN class that defaults with Xavier Init
			uses numpy at its core
		"""
		self._input_size = opts['input_size']
		#only supports cross entropy 
		#and MSE = mean squared error
		# CE = cross entropy
		self._loss_fn = opts['loss']
		#each one is dict:
		#{'W'Matrix,'a':string}
		#we use the most recent matrix shape
		#to define the next matrix shape
		self._layers = []
		#A,Z
		self._fwd_trace = []

	def Relu(self,X):
		switches = X>0
		return X*switches,switches

	def FC(self,h,activation='relu'):
		#initialize matrix for this layer
		#use previous matrix shape if no matrix exists
		#use input size
		prev_h = self._input_size

		if self._layers:
			prev_h = self._layers[-1]['W'].shape[0]

		#Xavier init
		#W = (np.random.rand(h,prev_h)-0.5)*math.sqrt(6/(h+prev_h))
		if activation is 'relu':
			#kaiming
			W = np.random.randn(h,prev_h)*math.sqrt(2/prev_h)
		else:
			#almost xavier
			W = 2*(np.random.rand(h,prev_h)-0.5)*math.sqrt(3/prev_h)
		b = np.zeros((h,1))

		self._layers.append({
			'type':'FC',
			'W':W,
			'b':b,
			'a':activation
		})

		#for train wrecks :)
		return self

	def Fwd(self,X):
		#add X as the A0 activation
		try:
			self._fwd_trace[0]['A'] = X
		except:
			self._fwd_trace.append({'A':X})

		fwd_prop = X
		for layer in self._layers:
			if layer['type'] == 'softmax':
				continue
			trace_item = {}
			Z = layer['W']@fwd_prop + layer['b']
			trace_item['Z'] = Z

			if layer['a'] is 'relu':
				A,switches = self.Relu(Z)
				trace_item['S'] = switches
			elif layer['a'] is 'sigmoid':
				A = 1/(1+np.exp(-Z))
			elif layer['a'] is 'rswitch':
				"""
					apply a random switch to 
					matrix maybe it acts as some
					sort of regularization like drop out?
				"""
				print('rswitch has issues between layers with different sizes')
				exit()
				switches = np.random.rand(*X.shape) > 0.1
				trace_item['S'] = switches
				A = X*switches
			else:
				#during back prop we'll treat none
				#as an identity
				A = Z

			#we might not want an activation
			#eg softmax or to test is linear
			#networks do better than reduced equivalent
			#hmmm........
			#note if no activation we treat it as identity map
			trace_item['A'] = A
			fwd_prop = A
			#book keeping
			self._fwd_trace.append(trace_item)

	def _Softmax(self):
		trace_item = {}
		#get last layer Z
		last_Z = self._fwd_trace[-1]['Z']
		last_Z = np.exp(last_Z)
		softmax= last_Z/np.sum(last_Z,axis=0)
		return softmax

	def Bwd(self,targets,lr):
		"""
			performs gradient descent with the given learning rate
			This allows for easy on the go learning rate adaptation
			note this does fully vectorized back propagation
			using the matrix back-propagation equations
			if you're curious abou the derivation see 
			https://rojasinate.com/documents/backprop.pdf

			TODO: use nestorov momentum
		"""
		m = targets.shape[1]
		#initialize back prop
		dZ = None
		if self._loss_fn is 'MSE':
			tmp = self._fwd_trace[-1]['A'] - targets
			loss = np.sum(tmp**2)/m
			dA = tmp
		elif self._loss_fn is 'SMCE':
			#do cross entorpy with softmax
			y_hat = self._Softmax()
			ones = np.ones_like(targets)
			loss = np.sum(-targets*np.log(y_hat))/m
			dA = self._fwd_trace[-1]['A'] - targets

		# > 1 because the first "layer" is the input layer
		while len(self._fwd_trace) > 1:
			
			fwd_data = self._fwd_trace.pop()
			#get params using the length of the fwd_data
			#as the index. pre popping makes the length the correct
			#index :) also need to subtract 1 to account for the A0 = X
			# "layer"
			layer = self._layers[len(self._fwd_trace)-1]
			if layer['type'] == 'FC':
				if layer['a'] == 'relu' or layer['a'] == 'rswitch':
					dZ = dA*fwd_data['S']
				elif layer['a'] is None:
					dZ = dA

				dA = layer['W'].T@dZ

			#soft max has no params
			if layer['type']!='softmax':
				layer['W'] -= lr*(1/m)*dZ@np.transpose(self._fwd_trace[-1]['A'])
				layer['b'] -= lr*(1/m)*np.sum(dZ,axis=1,keepdims=True)
		
		return loss

	def Fit(self,X,y,lr):
		model.Fwd(X)
		return model.Bwd(y,lr)

	def log_trace(self):
		for t in self._fwd_trace:
			print(t)
			print('---------------------------------------')

	def predict(self,X):
		self.Fwd(X)
		y_hat = self._fwd_trace[-1]['A']
		#clear out fwd trace
		self._fwd_trace = []
		return y_hat

def Y(X):
	return 4*X**3 - X**2 + 2*X +3 - 0.5*X**4

batch_size = 150
num_features = 4
#random permutation
prm = np.random.permutation(batch_size)
data = datasets.load_iris()
X = data.data.T[:num_features,prm] # we only take the first two features.
#center data
X = normalize(X,ax=1)
y = data.target.T[prm]
#one hot
num_classes = 3
y_oh = np.eye(num_classes)[y].T


split = 110
X_train = X[:,:split]
y_train = y_oh[:,:split]
print(X_train.shape,y_train.shape)
X_val = X[:,split:]
#observe the indices are different for y vs y_oh
y_val = y[split:]
print(X_val.shape,y_val.shape)

model = FFNN({
	'input_size':num_features,
	'loss':'SMCE'
})\
.FC(4)\
.FC(4)\
.FC(4)\
.FC(3)\
.FC(num_classes,None)

lr = 0.005
EPOCHS = 20000
loss_trace = []
for epoch in range(EPOCHS):
	loss = model.Fit(X_train,y_train,lr)
	loss_trace.append(loss)
	if epoch % 100 ==0:
		print('epoch',epoch,loss)

y_hat = np.argmax(model.predict(X_val),axis=0)
print(y_val[:5])
print(y_hat[:5])
accuracy = np.sum(y_val==y_hat)/y_val.shape[0]
print('accuracy:',accuracy)
"""
model_size = 50
X_model = (np.random.rand(num_features,model_size)-0.5)*2
y_hat = model.predict(X_test)
y_test = np.sum(Y(X_test),axis = 0,keepdims=True)

plt.plot([i for i in range(1,test_size+1)],y_hat[0,:])
plt.plot([i for i in range(1,test_size+1)],y_test[0,:])
plt.show()
"""
plt.plot(loss_trace)
plt.show()
