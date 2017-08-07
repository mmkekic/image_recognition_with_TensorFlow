#!/usr/bin/env python
import tensorflow as tf
from  dataset import *
import sys
import time

data_path='./fcbdata/'
data=DataSet(data_path)
n_clas=len(data.cat)
batch_size=64
im_size=data.im_size

x=tf.placeholder('float',[None,im_size,im_size,3])
y_=tf.placeholder('float',[None,n_clas])

def weight_variable(shape):
    #initialize network based on  He et al
    in_dim=np.prod(shape[:-1])
    initial = tf.truncated_normal(shape,stddev=np.sqrt(2./in_dim))
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x,size):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                          strides=[1, 2, 2, 1], padding='VALID')

weights={
    'h1': weight_variable([5, 5, 3, 32]),
    'h2': weight_variable([3, 3, 32, 64]),
    'h3': weight_variable([3, 3, 64, 128]),
    'fc1': weight_variable([7*7*128, 1024]),
    'fc2': weight_variable([1024, 3])
}

biases={
    'h1':  bias_variable([32]),
    'h2':  bias_variable([64]),
    'h3':  bias_variable([128]),
    'fc1':  bias_variable([1024]),
    'fc2': bias_variable([3])
}

keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)

def conv_neural_net(x):
    x_image = tf.reshape(x, [-1, im_size,im_size,3])    
    #conv1
    h_conv1= conv2d(x_image, weights['h1'])
    #relu
    h_conv1 = tf.nn.relu(h_conv1 + biases['h1'])
    #maxpool1
    h_pool1 = max_pool_2x2(h_conv1,2)
    #conv2
    h_conv2= conv2d(h_pool1,  weights['h2'])
    #relu
    h_conv2 = tf.nn.relu(h_conv2 + biases['h2'])
    #maxpool2
    h_pool2 = max_pool_2x2(h_conv2,2)
    #conv3
    h_conv3 = conv2d(h_pool2,  weights['h3'])
    #relu
    h_conv3=tf.nn.relu(h_conv3+ biases['h3'])
    #maxpool3
    h_pool3 = max_pool_2x2(h_conv3,3)
    h_pool3_flat = tf.reshape(h_pool3, [-1, 7*7*128])
    #fully connected1
    h_fc1=tf.matmul(h_pool3_flat, weights['fc1']) +  biases['fc1']
    #relu
    h_fc1 = tf.nn.relu(h_fc1)
    #dropout (training only)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    #fully conected2
    y_conv =  tf.matmul(h_fc1_drop,  weights['fc2']) +  biases['fc2']
    return y_conv

def train_model(x):
    prediction= conv_neural_net(x)
    #add l2 weight regularizator
    loss2=tf.add_n([tf.nn.l2_loss(weights[w]) for w in weights])*1e-3
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prediction)
                                   +loss2)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #we keep track of both  best accuracy and loss of the validation set
    best_acc=0.85
    best_loss=2.
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step=1e-3
        while (data.epoch_counts<500) :
            X,y =   data.next_batch(batch_size)
            optimizer.run(feed_dict={x:X, y_: y, keep_prob: 0.5,learning_rate : step})
            #print the status whenever epoch ends, ie batch_counts resets
            if data.batch_counts == 0:
                train_loss = cross_entropy.eval(feed_dict={
                    x:X , y_:y, keep_prob: 1.0})
                valid_loss = cross_entropy.eval(feed_dict={
                    x: data.valid_image()[0],y_: data.valid_image()[1],
                    keep_prob: 1.0})
                train_acc = accuracy.eval(feed_dict={
                    x:X , y_:y, keep_prob: 1.0})
                valid_acc= accuracy.eval(feed_dict={
                    x: data.valid_image()[0],y_: data.valid_image()[1],
                    keep_prob: 1.0})
                print('epoch %d , training loss %g, training accuracy %g' % (data.epoch_counts, train_loss, train_acc))
                print('validation loss %g, validation accuracy %g' % (valid_loss, valid_acc))
                #update best value accuracy and best loss
                if (valid_acc>best_acc) and (valid_loss<best_loss):
                    print('********************************************')
                    print('new best validation loss :)  %g' % (valid_loss))
                    print('********************************************')
                    saver.save(sess,'my_model.cktp')
                    best_acc=valid_acc
                    best_loss=valid_loss
                    last_save=data.epoch_counts
                #if validation accuracy and validation lossdidnt improve the last 50 epoch,
                #reload the model and reduce the step 
                try:
                    if (data.epoch_counts-last_save)>=50:
                        print('********************************************')
                        print ('Lets decrease the step and go back to the best validation point...')
                        step=step/10.
                        last_save=data.epoch_counts
                        saver = tf.train.import_meta_graph('./my_model.cktp.meta')
                        saver.restore(sess, './my_model.cktp')
                        valid_loss= cross_entropy.eval(feed_dict={
                            x: data.valid_image()[0],y_: data.valid_image()[1],
                            keep_prob: 1.0})
                        print(' validation loss %g' % ( valid_loss))
                        print('********************************************')
                        continue
                except:
                    pass
        print('Training finished!')
        
def test_model():
    prediction =conv_neural_net(x)
    #index of the highest category probability, used to print prediction
    predict_indx=tf.argmax(prediction, 1)
    correct_prediction = tf.equal(predict_indx, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()                
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, './my_model.cktp')
        X_t,y_t=data.test_image()
        test_acc= accuracy.eval(feed_dict={
            x: X_t,
            y_: y_t, keep_prob: 1.0})
        print('********************************************')
        print('Accuracy on the test data %g '%(test_acc))
        print('Lets look at 10 examples and compare with our prediction:')
        pred_=predict_indx.eval(feed_dict={
            x:X_t,
            y_:y_t, keep_prob: 1.0})
        for i in range(20):
            print(data.test[i],data.cat[pred_[i]])
        
if len(sys.argv)==1:
    t0=time.clock()
    train_model(x)
    dt=(time.clock()-t0)/60.
    print ('Training time %g'%(dt))
elif (sys.argv[1]=='test'):
    try:         
        test_model()
    except IOError:
        print ('There is no saved model, you need to train it first!')
else:
    print ('\'test\' is the only command-line argument that can be used.')
