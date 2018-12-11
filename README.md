top 7% solution for kaggle Quick, Draw! Doodle Recognition Challenge

### Why Failed
Let's talk about the failure first.
1. I didn't learn how to extend batchsize by freeze the bottom layers of CNN until the end of this competition which seems
to be a key skill to improve a single model.
2. I didn't solve the GPU memory limitation when increase the resolution to 512 as I've got only one 1080. It seems the skill above
can handle it, too. I will try it next time. As this time, I sticked to 71x71 with batch size 256.
3. I have not learnt the proper way to ensemble the results of CNNs so it was just weighted average. The [best method](https://www.kaggle.com/c/cdiscount-image-classification-challenge/discussion/45733) 
comes out after the competition ends.
4. I've got a crying baby and I feel it is not right to team up with others. Actually I didn't think I can get a medal this time because lacking of sleep.

### keras is losing in pretrained models?
I started with keras pretrained CNNs and found the best is xception.
But when I migrate to pytorch with se CNNs, I just got much better results.
Although the newest paper said there is no actual difference if start with pretrained models from imagenet or not,
but I think it may not be right in kaggle competitions.

### RNN works in simplfied data
It is a surprise that I found simple RNN with simplfied draw data can get a very good result, 0.922 in public LB.
But same model can not learn anything with detailed original draw data. Maybe it is because the original raw data is 
too sparse.

### better image better results
I think the key to this competition is how to encoding more information from the draw sequence into a 3 channel image.
It is absolutely important to use the raw data since only they can provide time scaled information.
I got my best single model with 2nd channel in draw time and 3rd channel in pause time.
But I failed to encode the attention map which was mentioned by the Grand Master.

Still a lot to learn. Hope my baby will cry less as he grows.
