@ECHO OFF

call conda activate cat2dog
echo Testing classifier...
python test_GDA.py
PAUSE