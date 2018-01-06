# kaggle-toxic-comments
[Kaggle "Toxic Comments" classification competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)



### Input  
Text data containing toxic comments, where different classes of "toxicity" have been labelled<br>
This is a multi-label task with input records belonging to one, multiple, or zero classes

### Output  
Probability of observation belonging to each class

### Evaluation
As observations can have multiple labels, Log-loss is calculated for each class, and then averaged across all classes
