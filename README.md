# Quora-Question-Pairs Task

## Info

* **quoraDataPrePreprocessing.ipynb** - python Ipynb for data preprocessing
* **quoraLSTM.ipynb** - iTorch Ipynb for training and validation of of deep network
* **dataset** - put trainData and valData in hdf5 format (update quoraLSTM.ipynb respective option argument)
* **trainingStats** - Contain training stats return by quoraLSTM.ipynb
* **models** - contain pretrained model on different hyperparameter setting
* **publishedDocs** - contain training and validation data for pretrained models

### Options for training and validation
```
opt = {
    useGPU = 1,
    loadModel = 0,
    loadModelPath = "models/quoraNetComp_batchSize_512_Lr_3.125e-05_learningRateDecay_0_optimizer_sgd_weightDecay_0.0005_stepDecayStep_5_epoch_20.t7",
    
    batchSize = 512, --IT should be less.. used to make it faster. good range 8-128 (2^i)
    -- The problem with large batch sizes are “sharp” local minima which leads to overfitting. 
    -- with large batch size the learning (Not ephoc time) will be slow
    --Larger batch sizes result in faster progress in training, but don't always converge as fast. 
    seqLen = 55,  --Assumed maximum words in sentance
    wordVecSize = 50,
    lstmOutputSize = 400,  --dont change

    useDropout = 1,
    dropoutProbability = 0.5,
    useBN = 1,
    useLeakyRelu = 0,
    leakyReluSlope = 0.2,

    useLoss = "SoftMax", -- "SoftMax" or "SVM" -- TAKE CARE LABEL SHOULD BE CHANGED TO 1 and -1 for "SVM"                      
    svmMargin = 1,
    optimizer = "adam",  --"sgd" or "adam"
    
    trainDataPath = "/dataset/train10000to60000.h5py",
    evalDataPath = "/dataset/train0to10000.h5py",  
    trainDataSize = 44020,
    valDataSize = 8793,  --Check
    maxTrainDataSize  = 50000,    -- Maximum to run while training
    maxValDataSize = 50000,  
    
    learningRate = 0.001,
    learningRateDecay = 0,--1e-5,  -- 1/t decay.. Set to 0 if setting step decay to true
    weightDecay = 0.0005,     -- l2 regularizer
    momentum = 0.9,
    nesterov = true,
    beta1 = 0.9,
    beta2 = 0.999,
    epsilon = 1e-8,
    stepDecay = false,
    stepDecayRate = 0.5,
    stepDecayStep = 5,
    
    nEpochs = 60,
    validateAtEach  = 1,
    saveAtEach = 15,
    savePath ="models/quoraNetComp",
    trainingStatsSavePath = "trainingStats/quoraNetComp",
    
    
    wieghtIntalization = "" -- "yaanLeCun" , "Xavier" or "KaimingHe"  (NOT IMPLEMENTED FOR NOW)
            
}

```


### Model

q1({w1,w2,w3...}) --> many-to-one-LSTM-- <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; --concatenate -- {FC --BN - RELU or LeakyRELU--Drop}*3 -- FC(1) -- {sigmoid--softmaxLoss} or SVM <br/> 
q2({w1,w2,w3..}) --> many-to-one-LSTM --
<br/>
<br/>
<br/>
The best validation accuracy I got is 0.745. (under constrained settings)


## Fix for problem with sharing parameters of LSTMs 

If the following is used for sharing parameters of two LSTM (Note: the same is mentioned in [torch documentation](https://github.com/torch/nn/blob/master/doc/module.md#sharemlps1s2sn) )<br/>
```
rnnSen2 = rnnSen1:clone('weight','bias');
```

Here optim will give error in running adam or sgd, As both networks share same weights, but weights-gradients are different. the weights-gradients will be double of the available wiegths. <br/>

Instead use<br/>
```
rnnSen2 = rnnSen1:clone('weight','bias','gradWeight','gradBias');
```

