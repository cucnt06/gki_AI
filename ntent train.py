[1mdiff --cc train.py[m
[1mindex 6f01f31,411e63a..0000000[m
[1m--- a/train.py[m
[1m+++ b/train.py[m
[36m@@@ -12,6 -12,53 +12,56 @@@[m [mfrom losses import regression_loss, dig[m
  from torch import movedim[m
  [m
  [m
[32m++<<<<<<< HEAD[m
[32m++=======[m
[32m+ """[m
[32m+ ##################[m
[32m+ ### QUESTION 1 ###[m
[32m+ ##################[m
[32m+ """[m
[32m+ [m
[32m+ [m
[32m+ def train_perceptron(model, dataset):[m
[32m+     with no_grad():[m
[32m+         dataloader = DataLoader(dataset, batch_size=1, shuffle=True)[m
[32m+ [m
[32m+         while True:[m
[32m+             mistakes = 0[m
[32m+ [m
[32m+             for batch in dataloader:[m
[32m+                 x = batch["x"][0][m
[32m+                 label = batch["label"].item()[m
[32m+ [m
[32m+                 prediction = model.get_prediction(x)[m
[32m+ [m
[32m+                 if prediction != label:[m
[32m+                     model.w += label * x[m
[32m+                     mistakes += 1[m
[32m+ [m
[32m+             if mistakes == 0:[m
[32m+                 break[m
[32m+ [m
[32m+ [m
[32m+ [m
[32m+ def train_regression(model, dataset):[m
[32m+     """[m
[32m+     Trains the model.[m
[32m+ [m
[32m+     In order to create batches, create a DataLoader object and pass in `dataset` as well as your required [m
[32m+     batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader[m
[32m+ [m
[32m+     Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label[m
[32m+     is the item we need to predict based off of its features.[m
[32m+ [m
[32m+     Inputs:[m
[32m+         model: Pytorch model to use[m
[32m+         dataset: a PyTorch dataset object containing data to be trained on[m
[32m+         [m
[32m+     """[m
[32m+     "*** YOUR CODE HERE ***"[m
[32m+ [m
[32m+ [m
[32m++>>>>>>> q1[m
  def train_digitclassifier(model, dataset):[m
      """[m
      Trains the model.[m
