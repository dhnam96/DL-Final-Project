# DL-Final-Project
DL Final Project

Approach 1: preprocess_org.py, assignment_org.py
Approach 2: preprocess.py, assignment.py

Here, we specify how to run the second approach but the first approach should be in the repository for reference

1) clone this repository 
2) download the celebA dataset from /course/cs1470/asgn/dcgan/celebA.tar.gz
3) extract the celebA dataset 

To run the code, make sure you're in the CS147 virtual environment:

python3 assignment.py --mode train --num-epochs 100

python3 assignment.py --mode test

python3 assignment.py --mode train_completion --num-epochs 50

python3 assignment.py --mode test_completion

make sure that the celebA dataset lives in the correct directory and change the data home path in line 381 in main() of assignment.py if path is different

