DBN_Classification
====================

 
 Example of DBN classification using python, tensorflow and using malware dataset
 

# Overview

    * Use code from existing books and some revising
    * Evaluate classification performance in malware detection
    * DBN implementation using Python and tensor flow
    * Check the difference between each method
    * Simple usage example


# Detailed explanation


<img width="350" alt="dbn_pre" src="https://user-images.githubusercontent.com/37811577/55939582-171bb200-5c79-11e9-9d9b-cff71e41922d.png">


DBN as pre-train method

* python code 
> 1. Using Python, we show the detailed implementation and simple usage of the DBN algorithm.
> 2. Put binaryalphadigs.mat data on input node and perform reconstruction.
> 3. Greedy algorithm and wake-sleep algorithm are used in learning process.


* tensorflow code
> 1. Evaluate classification performance using malware dataset.
> 2. We used 1224 files as datasets, using 779 malware files and 445 benign files.
> 3. Malware files are download from Web sites and Benign files are from Window file directory.
> 4. We extracted the opcode using the dataset above and used it as a feature vector.
> 5. Run this code 30 times, so we selected and used the value of random seeds differently for reproduction of same result.
> 6. We did not upload code such as AUC calculation.
> 7. The example below shows the code that changes such as the initialization method, activation function.
> 8. The following figure is the simple structure of this classifier code.
> 9. ###### This code is not fully organized.

<img width="284" alt="dbn_con" src="https://user-images.githubusercontent.com/37811577/55939558-04a17880-5c79-11e9-8b0b-096b3a89b1e7.png">

  


# Origin

* python code : Machine Learning: An Algorithmic Perspective, Second Edition by Stephen Marsland

![algo](https://user-images.githubusercontent.com/37811577/55819714-51c80200-5b34-11e9-8e02-86b5ba9644d5.jpg)

* tensorflow code : TensorFlow 1.x Deep Learning Cookbook by Antonio Gulli, Amita Kapoor

![purple](https://user-images.githubusercontent.com/37811577/55819740-5f7d8780-5b34-11e9-97d1-206b6d9f6dac.jpg)

# reference

* RBM implementation : https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

* Xavier Glorot, Yoshua Bengio, “Understanding the difficulty of training deep feedforward neural networks”, Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 249-256, March 2010

* Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, “Delving Deep into Rectiﬁer”, Proceedings of the 2015 IEEE International Conference on Computer Vision (ICCV), pp, 1026-1034 , December 07 - 13, 2015


# Example

* python : 

<code>
  
    import dbn
    
    dbn.test_dbn_digs()
</code>



* tensorflow import : 

<code>
 
    opcode_count = pd.read_csv(os.path.join("Opcode_1gram_count.csv"), sep='\t', index_col=0)
    ...
    
    for i in repeat_list:
      X_train, X_test, y_train, y_test = train_test_split(np_data_float, one_label, test_size=0.2, random_state=i)
      X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=i)
    
    ...
    import RBM
    import sigmoid_DBN
    ....
    
</code>


* tensorflow build DBN :
<code>
  
    for i, size in enumerate(RBM_hidden_sizes):
      print('RBM: ',i,' ',input_size,'->',size)
      rbm_list.append(RBM.RBM(input_size,size))  #2304,1500   1500, 700   700,400
      input_size = size
      
    for rbm in rbm_list:
      print('New RBM:')
    
    with tf.Session() as sess:
      sess.run(init)
      rbm.set_session(sess)
      err = rbm.fit(inpX, 1)         # input, epoch
      inpX_n = rbm.rbm_output(inpX)    
      print(inpX_n.shape)
      inpX = inpX_n
</code>


* example of changing activate fuction and initialization method : 
<code>
      
    self._a[3] = layers.fully_connected(self._a[2], self._sizes[-1] , activation_fn=tf.nn.sigmoid, biases_initializer=tf.zeros_initializer)
    
    change like this
   
    self._a[3] = layers.fully_connected(self._a[2], self._sizes[-1],weights_initializer=tf.contrib.layers.variance_scaling_initializer(uniform=True) , activation_fn=tf.nn.relu, biases_initializer=tf.zeros_initializer)
      
</code>



# output :


* python 

origin
![algo_origin](https://user-images.githubusercontent.com/37811577/55940952-29e3b600-5c7c-11e9-9866-4a2df57beb9d.png)

reconstructed
![algo_recon](https://user-images.githubusercontent.com/37811577/55941008-42ec6700-5c7c-11e9-94cd-7ea7ecc05d79.png)


  
 
 * tensorflow
      
       epoch0/100Training Accuracy : 0.7522349936143039 Validation Accuracy : 0.7397959183673469
       epoch1/100Training Accuracy : 0.8850574712643678 Validation Accuracy : 0.8724489795918368
       epoch2/100Training Accuracy : 0.9118773946360154 Validation Accuracy : 0.8928571428571429
       epoch3/100Training Accuracy : 0.9348659003831418 Validation Accuracy : 0.9183673469387755
       epoch4/100Training Accuracy : 0.9412515964240102 Validation Accuracy : 0.923469387755102
       epoch5/100Training Accuracy : 0.9425287356321839 Validation Accuracy : 0.9285714285714286
       ....
       ...
       ..
       .
       epoch96/100Training Accuracy : 0.9872286079182631 Validation Accuracy : 0.9744897959183674
       epoch97/100Training Accuracy : 0.9872286079182631 Validation Accuracy : 0.9744897959183674
       epoch98/100Training Accuracy : 0.9872286079182631 Validation Accuracy : 0.9744897959183674
       epoch99/100Training Accuracy : 0.9885057471264368 Validation Accuracy : 0.9744897959183674
       Accuracy%: 98.77551198005676
       end_time :  5.727693557739258
       
 
 
 
 
 
 https://www.ijitee.org/wp-content/uploads/papers/v8i4s2/D1S0014028419.pdf 



