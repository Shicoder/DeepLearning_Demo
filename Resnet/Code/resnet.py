import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(2142)
from subprocess import check_output
print(check_output(["ls", "../Data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Convolution2D, Input,Activation, ZeroPadding2D, MaxPooling2D, Flatten, merge
from keras.optimizers import SGD
from keras.objectives import sparse_categorical_crossentropy as scc
#using Tensorflow backend
train=pd.read_csv("../Data/train.csv")
test=pd.read_csv("../Data/test.csv")
#let's separate stuff to make it more manageable
y_train=train['label']
train.drop(['label'],axis=1,inplace=True)
######################Preprocessing involved##########
#########################1.Feature Scaling(division by 255)#####
##########################2.Per Image Normalization#############
x_train=train.values.astype('float32')/255
x_test=test.values.astype('float32')/255
# below is a custom code for per image normalization.
# It is faster than looping

# the constant term is as Advised by Andrew Ng in his UFLDL Tutorials

def per_image_normalization(X, constant=10.0, copy=True):
    if copy:
        X_res = X.copy()
    else:
        X_res = X

    means = np.mean(X, axis=1)
    variances = np.var(X, axis=1) + constant
    X_res = (X_res.T - means).T
    X_res = (X_res.T / np.sqrt(variances)).T
    return X_res

x_train = per_image_normalization(x_train)
x_test = per_image_normalization(x_test)
####Now, we'll reshape the input to the shape 1, 28, 28
#I am running theano. An epoch on tensorflow was taking 48 seconds!
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
###########################################################################################
##Now let's give a brief about Residual Nets, or Simply ResNets
##In this notebook, we'll look at a 15 layer Residual Network (counting
# only those layers which have trainable parameters. Activation, Merge,
# and Dropout Layers are not counted)

##Salient features of the Resnet implemented
#1.As originally proposed, no MaxPooling Layers are used. Down Sampling is done by varying the strides and kernels of the Convolutional2D layers only
# 2.Even though the network is so deep, the number of parameters is very small (as you'll notice later)
# 23We use only 2 residual blocks.
# 4.The original proposal didn't use Dropout. The implemented model has Dropout between the final Dense Layers.
# 5.The layers are well labelled and it makes it a little easier to see what's actually going on
# 6.NO HYPERPARAMETERS WERE TUNED, PERIOD. Every parameter, including the number of Feature maps at every Convolution, was arbitrarily chosen. Therefore, there is a lot of space for hyperparam Tuning.
# 7.The implementation has been done using the Keras Functional API.
###############################################################################
# lets get to it and define the function that will make up the network

def get_resnet():
    # In order to make things less confusing, all layers have been declared first, and then used

    # declaration of layers
    input_img = Input((1, 28, 28), name='input_layer')
    zeroPad1 = ZeroPadding2D((1, 1), name='zeroPad1', dim_ordering='th')
    zeroPad1_2 = ZeroPadding2D((1, 1), name='zeroPad1_2', dim_ordering='th')
    layer1 = Convolution2D(6, 3, 3, subsample=(2, 2), init='he_uniform', name='major_conv', dim_ordering='th')
    layer1_2 = Convolution2D(16, 3, 3, subsample=(2, 2), init='he_uniform', name='major_conv2', dim_ordering='th')
    zeroPad2 = ZeroPadding2D((1, 1), name='zeroPad2', dim_ordering='th')
    zeroPad2_2 = ZeroPadding2D((1, 1), name='zeroPad2_2', dim_ordering='th')
    layer2 = Convolution2D(6, 3, 3, subsample=(1, 1), init='he_uniform', name='l1_conv', dim_ordering='th')
    layer2_2 = Convolution2D(16, 3, 3, subsample=(1, 1), init='he_uniform', name='l1_conv2', dim_ordering='th')

    zeroPad3 = ZeroPadding2D((1, 1), name='zeroPad3', dim_ordering='th')
    zeroPad3_2 = ZeroPadding2D((1, 1), name='zeroPad3_2', dim_ordering='th')
    layer3 = Convolution2D(6, 3, 3, subsample=(1, 1), init='he_uniform', name='l2_conv', dim_ordering='th')
    layer3_2 = Convolution2D(16, 3, 3, subsample=(1, 1), init='he_uniform', name='l2_conv2', dim_ordering='th')

    layer4 = Dense(64, activation='relu', init='he_uniform', name='dense1')
    layer5 = Dense(16, activation='relu', init='he_uniform', name='dense2')

    final = Dense(10, activation='softmax', init='he_uniform', name='classifier')

    # declaration completed

    first = zeroPad1(input_img)
    second = layer1(first)
    second = BatchNormalization(0, axis=1, name='major_bn')(second)
    second = Activation('relu', name='major_act')(second)

    third = zeroPad2(second)
    third = layer2(third)
    third = BatchNormalization(0, axis=1, name='l1_bn')(third)
    third = Activation('relu', name='l1_act')(third)

    third = zeroPad3(third)
    third = layer3(third)
    third = BatchNormalization(0, axis=1, name='l1_bn2')(third)
    third = Activation('relu', name='l1_act2')(third)

    res = merge([third, second], mode='sum', name='res')

    first2 = zeroPad1_2(res)
    second2 = layer1_2(first2)
    second2 = BatchNormalization(0, axis=1, name='major_bn2')(second2)
    second2 = Activation('relu', name='major_act2')(second2)

    third2 = zeroPad2_2(second2)
    third2 = layer2_2(third2)
    third2 = BatchNormalization(0, axis=1, name='l2_bn')(third2)
    third2 = Activation('relu', name='l2_act')(third2)

    third2 = zeroPad3_2(third2)
    third2 = layer3_2(third2)
    third2 = BatchNormalization(0, axis=1, name='l2_bn2')(third2)
    third2 = Activation('relu', name='l2_act2')(third2)

    res2 = merge([third2, second2], mode='sum', name='res2')

    res2 = Flatten()(res2)

    res2 = layer4(res2)
    res2 = Dropout(0.4, name='dropout1')(res2)
    res2 = layer5(res2)
    res2 = Dropout(0.4, name='dropout2')(res2)
    res2 = final(res2)
    model = Model(input=input_img, output=res2)

    sgd = SGD(decay=0., lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss=scc, optimizer=sgd, metrics=['accuracy'])
    return model

res = get_resnet()
################################################################
               #Let's checkout the network a bit first.
            #First, we'll print a summary
################################################################
res.summary()
##########################################################################
# As you can see, for a network this big, the number of parameters is TINY!!!!!!
# This will greatly aid in avoiding overfitting!
# Let's visualize the network next.. This will be a big, scrollable picture !!!! ( no pydot ' \
# here. try on ur own. I'll post the image in comments)
###########################################################################
#from IPython.display import SVG
#from keras.utils.visualize_util import model_to_dot

#SVG(model_to_dot(res).create(prog='dot', format='svg'))
###########################################################################
# I wish I had a better way to visualize the network. But nonetheless, this looks SICK!
# Now it's time to put the network to test. (note, on theano, can take 1-3 mins to compile)
#############################################################################
# we'll use a simple cross validation split of 5%, because any other cross validation scheme doesn't make sense
history = res.fit(x_train, y_train, validation_split=0.05, verbose=2, nb_epoch=40, batch_size=32)
#############################################################################
# I have only trained the model for a mere 1 epochs because
# 1.Kaggle notebooks take toooooooo long to run the model.
# 2.On my laptop with 8gb ram and gt740m 2gb graphics card, and Theano backend, it takes merely 13 seconds per epoch.
###############################################################################
# NOTE : I will be writing about the results obtained on training for 30 epochs.


# Anyways, if there is a lot happening here
# Consideri 1.The max validation accuracy achived is 0.9886.ng our rather naive validation scheme (meh) and the final results obtained on the leader board (0.98500), this might not seem somethin to be too excited about. However, upon comparing the validation scores with the (max) training accuracy achieved by the model which is a mere 0.9449, one can easily see that the model is nowhere near convergence. (I dont know about you, but on seeing this, my jaw dropped to the floor).
# 2.Even MLP's easily overfit MNIST, whereas here, you can probably continue to train it for a lot, lot more epochs!
# 3.Did I mention how good the train accuracy vs validation accuracy setting is for a network that is THIS DEEP!!!!

# Let's visualize the training process ( plot attached separately in comments)

import matplotlib.pyplot as plt

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.legend(['train', 'val'])
#plt.show()
##############################################################
# Like I said, the model is no-where near convergence. (The plot is for 30 epochs
# trained on my system) MY advice:train it for atleast 50 more epochs with early stopping.

# LETS GET SOME PREDICTIONS!!!!
# Well..... Not so fast, there is a lil work here.The Keras functional API doesnt
# suppost predict_classes function, so we are gonna have to do this manually
################################################################3
# SUMMON sklearn's LabelBinarizer
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer().fit(y_train)
# lets get predictions now
preds = res.predict(x_test)
classes = (preds > 0.5).astype('int32')

# for those that dont know what happened, the above statement gave us binarized labels for each class
# this will give us labels as we need for submission
p = lb.inverse_transform(classes)
###################submission##############################
sub = pd.DataFrame()
ids = [x for x in range(1, 28001)]
sub['ImageId'] = ids
sub['Label'] = p
sub.to_csv('resne40t.csv', index=False)



##############################################
# ResNets are excellent if you can afford the memory requirements.
# Even though the network is so deep, the number of parameters stays relatively small, and so does training time.
# This notebook demonstrates us the reason WHY RESNETS PERFORM SO WELL ON problems like CIFAR10 (results in paper) without any data augmentation.
# Considering the small number of epochs trained and virtually no hyper parameter optimization at all (I selected kernel params, strides and sizes
# just to ensure that a valid model is created) I believe there is a lot of space for improvement here!!!!
# Some such possible places where improvements can be made are a. Using a different optimizer: Different learning rates, different algorithm altogether b. different kernel sizes, filter sizes and strides c. more layers (if you can, once you adjust kernel sizes) d. basically any type of hyperparameter tweaking e. regularization
# I said it before, and I say it again, the accuracy of the model on the training set is a mere 0.9467, and THIS AMOUNT OF UNDERFITTING has given us a leaderboard score of 0.98500. THIS IS FRICKIN INSANE!!!