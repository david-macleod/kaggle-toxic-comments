# kaggle-toxic-comments
Kaggle "Toxic Comments" classification competition 
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge


## INPUT 
Text data containing toxic comments, where different classes of "toxicity" have been labelled 
This is a multi-label task with input records belonging to one, multiple, or zero classes

## OUTPUT 
Probability of observation belonging to each class

## EVALUATION
As observations can have multiple labels, Log-loss is calculated for each class, and then averaged across all classes
