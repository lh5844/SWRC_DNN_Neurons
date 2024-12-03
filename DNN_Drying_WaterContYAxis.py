# 1.1 Import the primary libraries 
import tensorflow.compat.v1 as tf
import numpy
import numpy as np
import matplotlib.pyplot as plt
# 1.2 Import the secondary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 
from tensorflow.compat.v1  import set_random_seed 

tf.compat.v1.disable_eager_execution()

# Fix a random seed  to initialize random no. generator
# initialized random number generator to ensure that the results are REPRODUCIBLE
np.random.seed(1)
set_random_seed(2)
  
# 2. Define learning parameters i.e. learning rate (for gradient descent) & the number of steps (epochs)
learning_rate   = 0.01
learning_rate_ANN   = 0.01
momentum        = 0.1 #Optional --- used if using tensorflow "MomentumOptimizer" instead of "GradientDescentOptimizer"
training_epochs =5000
training_epochs_Art =5000

# 3. Import the soil Data (measured/observed suction and soil moisture)

soilData = pd.read_csv('Test_data/Drying1.csv')
X = soilData.iloc[:, 2:3].values # Soil suction (cm): This creates x as matrix instead of vector
X.shape
print(X)
y = soilData.iloc[:, 0:1].values #  Observed Soil moisture ($cm^3$/$cm^3$) : create y as matrix instead of vector
y_mean = soilData['Observed Water Content'].mean()
#y=y.ravel()
y.shape
print(y) 

# 4. Feature Scaling (Standardizing the data i.e. xnew=(x-mean)/std)
# Feature Scaling (Standardizing) is a process of transforming variables to have values in same range so that no variable is dominated by the other.
# tensorflow libraries do not apply feature scaling by default. Therefore, apply feature scaling to the X,Y dataset using sklearn libraries

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() # StandardScalar() is the class used for feature scaling from sklearn library.
sc_y = StandardScaler()
X = sc_X.fit_transform(X) # sc_x.fit_transform(X) transforms and fits X into scaled data
y = sc_y.fit_transform(y)

# 5. define training & test data:
# Typically, the data is subdivided into train and test data in the ration of 0.9 to 0.1, respectively.
# However, since the measured dataset is small and monotonic in nature, the same data is used to train and test the model
train_X = X
print("\ntrain_X=", train_X)
train_Y = y
 
test_X = X
print("\ntest_X=", test_X)
test_Y = y

# 7. define placeholders (input nodes of the graph)
#	 placeholders are where training samples (x,f(x)) are placed
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 8. setup the models: 8 DNN models
 
# DNN model generation helper function
def Deep_NN(x, weights, biases):
     
    # hidden layer #1 with RELU: layer_1=relu(W1*x+b1)
	# hidden layer #2 with RELU: layer_2=relu(W2*layer_1+b2)
	# hidden layer #3 with RELU: layer_3=relu(W3*layer_2+b3)
	# hidden layer #4 with RELU: layer_4=relu(W4*layer_3+b4)
	# hidden layer #5 with RELU: layer_5=relu(W5*layer_4+b5)
	# output layer #6 with linear: layer_6=linear(W_out*layer_5+b_out)
    reshaped_x = tf.reshape(x, [-1, 1])
    layer_1 = tf.add(tf.matmul(reshaped_x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
      
    # hidden layer #2 with RELU: layer_2=relu(W2*layer1+b2)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
	
	# hidden layer #3 with RELU: layer_2=relu(W2*layer1+b2)
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
	
	# hidden layer #4 with RELU: layer_2=relu(W2*layer1+b2)
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

	# hidden layer #4 with RELU: layer_2=relu(W2*layer1+b2)
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_5 = tf.nn.relu(layer_5)
	
    # output layer with linear activation (no RELUs!): out_layer=W_out*layer2+b_out
    out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
      
    # return the DNN model
    return out_layer

def Deep_NN_5HiddenLayers(x, weights5, biases5):
     
    # hidden layer #1 with RELU: layer_1=relu(W1*x+b1)
	# hidden layer #2 with RELU: layer_2=relu(W2*layer_1+b2)
	# hidden layer #3 with RELU: layer_3=relu(W3*layer_2+b3)
	# hidden layer #4 with RELU: layer_4=relu(W4*layer_3+b4)
	# hidden layer #5 with RELU: layer_5=relu(W5*layer_4+b5)
	# output layer #6 with linear: layer_out=linear(W_out*layer_5+b_out)
    reshaped_x = tf.reshape(x, [-1, 1])
    layer_1 = tf.add(tf.matmul(reshaped_x, weights5['h1']), biases5['b1'])
    layer_1 = tf.nn.relu(layer_1)
      
    # hidden layer #2 with RELU: layer_2=relu(W2*layer1+b2)
    layer_2 = tf.add(tf.matmul(layer_1, weights5['h2']), biases5['b2'])
    layer_2 = tf.nn.relu(layer_2)
	
	# hidden layer #3 with RELU: layer_2=relu(W2*layer1+b2)
    layer_3 = tf.add(tf.matmul(layer_2, weights5['h3']), biases5['b3'])
    layer_3 = tf.nn.relu(layer_3)
	
	# hidden layer #4 with RELU: layer_2=relu(W2*layer1+b2)
    layer_4 = tf.add(tf.matmul(layer_3, weights5['h4']), biases5['b4'])
    layer_4 = tf.nn.relu(layer_4)

	# hidden layer #4 with RELU: layer_2=relu(W2*layer1+b2)
    layer_5 = tf.add(tf.matmul(layer_4, weights5['h5']), biases5['b5'])
    layer_5 = tf.nn.relu(layer_5)
	
    # output layer with linear activation (no RELUs!): out_layer=W_out*layer2+b_out
    out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
      
    # return the DNN model
    return out_layer

	
# 9. Specify the number of neurons per layer i.e. neurons in input layer, hidden layer(s), and output layer
dim_in = 1
dim1 = 7
dim2 = 7
dim3 = 7
dim4 = 7
dim5 = 7
dim_out = 1
      
# 10. Create dictionaries to hold the weights & biases of all layers
weights = {
    'h1': tf.Variable(tf.random_normal([dim_in, dim1])),
    'h2': tf.Variable(tf.random_normal([dim1, dim2])),
	'h3': tf.Variable(tf.random_normal([dim2, dim3])),
	'h4': tf.Variable(tf.random_normal([dim3, dim4])),
	'h5': tf.Variable(tf.random_normal([dim4, dim5])),
    'out': tf.Variable(tf.random_normal([dim2, dim_out]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([dim1])),
    'b2': tf.Variable(tf.random_normal([dim2])),
	'b3': tf.Variable(tf.random_normal([dim3])),
	'b4': tf.Variable(tf.random_normal([dim4])),
	'b5': tf.Variable(tf.random_normal([dim5])),
    'out': tf.Variable(tf.random_normal([dim_out]))
}

weights5 = {
    'h1': tf.Variable(tf.random_normal([dim_in, dim1])),
    'h2': tf.Variable(tf.random_normal([dim1, dim2])),
	'h3': tf.Variable(tf.random_normal([dim2, dim3])),
	'h4': tf.Variable(tf.random_normal([dim3, dim4])),
	'h5': tf.Variable(tf.random_normal([dim4, dim5])),
    'out': tf.Variable(tf.random_normal([dim2, dim_out]))
}
biases5 = {
    'b1': tf.Variable(tf.random_normal([dim1])),
    'b2': tf.Variable(tf.random_normal([dim2])),
	'b3': tf.Variable(tf.random_normal([dim3])),
	'b4': tf.Variable(tf.random_normal([dim4])),
	'b5': tf.Variable(tf.random_normal([dim5])),
    'out': tf.Variable(tf.random_normal([dim_out]))
}

# 11. Now use helper function that was previously defined to generate a DNN/ANN models
	#ANN and DNN
pred = Deep_NN(X, weights, biases) 

	#5 Hidden Layers
pred_5Layers_DNN = Deep_NN_5HiddenLayers(X, weights5, biases5)

# 12. Define the DNN/ANN cost function that needs to be optimzed in order to get optimal weights and biases for the DNN layers:
# This is acieved by minimizing the sum of squared errors (SSE)
n_samples=len(test_X)
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

cost5 = tf.reduce_sum(tf.pow(pred_5Layers_DNN -Y, 2))/(2*n_samples) 

# 13. Generate optimizer node for gradient descent algorithm using tensorflow GradientDescentOptimizer() function
#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)  #Method 1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)		 #Method 2

optimizer5 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost5)
 
# 14. initialize the variables (to be optimized) & launch optimization 
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

init5 = tf.initialize_all_variables()
sess5 = tf.InteractiveSession()
sess5.run(init5) 


# 15. Kick off the training of the ANN/DNN models ...
print("Starting with training in Deep_NN...")
for epoch in range(training_epochs):
     # for all individual data samples we have ...  
    for (x, y) in zip(train_X, train_Y):
        # run a backprob step
        sess.run(optimizer, feed_dict={X: x, Y: y})        
    # display epoch nr
    if epoch % 20 == 0:
        c = sess.run(cost, feed_dict={X: test_X, Y:test_Y})
        print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(c))
print("Deep_NN Optimization Finished!")

print("Starting with training in Deep_NN_5HiddenLayers...")
for epoch in range(training_epochs):
     # for all individual data samples we have ...  
    for (x, y) in zip(train_X, train_Y):
        # run a backprob step
        sess5.run(optimizer5, feed_dict={X: x, Y: y})       
    # display epoch nr
    if epoch % 20 == 0:
        c5 = sess5.run(cost5, feed_dict={X: test_X, Y:test_Y})
        print("Epoch:", '%04d' % epoch, "cost=", "{:.9f}".format(c5))
print("Deep_NN_5HiddenLayers Optimization Finished!")
 
# 16. Display final costs
test_cost = sess.run(cost, feed_dict={X: test_X, Y: test_Y})
print("Test dataset DNN costs=", test_cost, "\n")

test_cost5 = sess5.run(cost5, feed_dict={X: test_X, Y: test_Y})
print("Test dataset DNN5 costs=", test_cost5, "\n")

# 17. Calculate the prediction by ANN/DNN models 
predicted_Y = numpy.zeros(n_samples)
for i in range(len(test_X)):
    y_value = sess.run(pred, feed_dict={X: test_X[i]})
    #print("i=",i, "x=",test_X[i], "f(x)=",y_value)
    predicted_Y[i] = y_value
predicted_Y = predicted_Y.reshape(-1, 1)
predicted_Y=sc_y.inverse_transform(predicted_Y)
#print(predicted_Y)

predicted_DNN5_Y = numpy.zeros(n_samples)
for i in range(len(test_X)):
    y_value5 = sess5.run(pred_5Layers_DNN, feed_dict={X: test_X[i]})
    #print("i=",i, "x=",test_X[i], "f(x)=",y_value)
    predicted_DNN5_Y[i] = y_value5

predicted_DNN5_Y = predicted_DNN5_Y.reshape(-1, 1)
predicted_DNN5_Y=sc_y.inverse_transform(predicted_DNN5_Y)
print(pd.DataFrame(predicted_DNN5_Y)) 
print(pd.DataFrame(predicted_DNN5_Y).dtypes)



# 18. Save predicted SWC

predicted_SWC=soilData.copy()
predicted_SWC['SWC_DNN']=pd.DataFrame(predicted_Y).iloc[:, 0:1].values
predicted_SWC['SWC_DNN_5HiddenLayers']=pd.DataFrame(predicted_DNN5_Y).iloc[:, 0:1].values

print(predicted_SWC)
predicted_SWC.to_csv('./Output_files/DNN_Layer5_predicted_Drying1_SWC_Epoch=5000.csv', encoding='utf-8', index=False)

# Willmot's index of agreement d1
d1_DNN = 1-(((soilData['Observed Water Content']-predicted_SWC['SWC_DNN']).sum())/(abs(soilData['Observed Water Content']-y_mean)+abs(predicted_SWC['SWC_DNN']-y_mean)).sum())
d1_DNN5 = 1-(((soilData['Observed Water Content']-predicted_SWC['SWC_DNN_5HiddenLayers']).sum())/(abs(soilData['Observed Water Content']-y_mean)+abs(predicted_SWC['SWC_DNN_5HiddenLayers']-y_mean)).sum())

print(': and d1_DNN='+str(d1_DNN)+ ': and d1_DNN5='+str(d1_DNN5))

# 19. plot measured/observed SWC/SWRC and ANN/DNN predicted SWC/SWRC

fig = plt.figure()
ax = plt.subplot(111)

plt.scatter(soilData.iloc[:, 0:1].values, soilData.iloc[:, 2:3].values, label='Observed Data, '+'N='+str(n_samples), facecolors='none', edgecolors='r')

plt.semilogy(predicted_DNN5_Y, soilData.iloc[:, 2:3].values, 'k^' ,label='DNN5: $R^2=$' + str(round(np.sqrt(r2_score(soilData.iloc[:, 0:1].values, predicted_DNN5_Y)),2))+ ', RMSE='+str(round(np.sqrt(mean_squared_error(soilData.iloc[:, 0:1].values, predicted_DNN5_Y)),3))  + '$cm^3$/$cm^3$'+', d1='+str(round(d1_DNN5,2)))

plt.xlabel('Soil Water Content ($cm^3$/$cm^3$)')
plt.ylabel('Suction (cm)')
#plt.xlim([0, 0.5])
plt.ylim([0, 100])
from matplotlib.ticker import LogLocator

#plt.legend(loc='lower left')
#plt.legend(loc='upper right')

ax.legend(ncol=2,handleheight=2.1, labelspacing=0.05, bbox_to_anchor=(0.5, -0.1), loc='upper center')


plt.tight_layout()
# 20. Saave the plots
plt.savefig('./Output_files/NeuralNetwork_Drying1_SWC_5Hidden-Layers_Epoch=5000.png', bbox_inches="tight")
plt.savefig('./Output_files/NeuralNetwork_Drying1_SWC_5Hidden-Layers_Epoch=5000.pdf', bbox_inches="tight")
plt.show()