This repository contains the solution to the task 1.4 of the third assignment of the course Introduction to Deep Learning (Leiden University).  
The goal is to predict the solution to addition/subtraction between two 3-digits (or less) numbers. The first number has to be positive. What makes this task not trivial is that the model has to generate the solution as a sequence of 4 images, each representing a digit, a - sign, or an empty space.

## Requirements
<ul>
  <li>numpy - pip install numpy</li>
  <li>numpy - pip install --upgrade tensorflow</li>
  <li>numpy - python -m pip install -U matplotlib</li>
  <li>numpy - pip install -U scikit-learn</li>
</ul>

## Train
`python --train PATH_1 --data PATH_2 --train_size 0.7 --pretrained PATH_3 --summary`  
Remove --pretrained for training the model from scratch.  
PATH_1: path where to store the weights after the training  
PATH_2: path to dataset  
PATH_3: path to pretrained weights  

## Run
Evaluate a single expression defined in --eval. The expression must have the form ddd?ddd, d=digit or whitespace and ?=+ or - (e.g. " 27+ &nbsp;5").  
`python --eval "640-200" --pretrained PATH_1 --eval_out PATH_2`  
PATH_1: path to pretrained weights  
PATH_2: path where to save the output images

## Examples
### 2 - 78 =
![](https://github.com/riccardomajellaro/Text2Image_AddSubSolver/blob/main/output/2-78.png)
### 555 + 500 =
![](https://github.com/riccardomajellaro/Text2Image_AddSubSolver/blob/main/output/555%2B500.png)
### 623 + 271 =
![](https://github.com/riccardomajellaro/Text2Image_AddSubSolver/blob/main/output/623%2B271.png)
### 640 - 200 =
![](https://github.com/riccardomajellaro/Text2Image_AddSubSolver/blob/main/output/640-200.png)
### 950 - 20 =
![](https://github.com/riccardomajellaro/Text2Image_AddSubSolver/blob/main/output/950-20.png)
### 999 + 700 =
![](https://github.com/riccardomajellaro/Text2Image_AddSubSolver/blob/main/output/999%2B700.png)
