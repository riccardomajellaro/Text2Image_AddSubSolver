This repository contains the solution to the task 1.4 of the third assignment of the course Introduction to Deep Learning (Leiden University).

## Train
`python --train PATH_1 --data PATH_2 --train_size 0.7 --pretrained PATH_3 --summary`  
Remove --pretrained for training the model from scratch.  
PATH_1: path where to store the weights after the training  
PATH_2: path to dataset  
PATH_3: path to pretrained weights  

## Run
Evaluate a single expression defined in --eval.  
`python --eval "640-200" --pretrained PATH_1 --eval_out PATH_2`  
PATH_1: path to pretrained weights  
PATH_2: path where to save the output images
