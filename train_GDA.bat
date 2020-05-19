@ECHO OFF

call conda activate cat2dog
echo Training classifier...
python train_GDA.py
PAUSE