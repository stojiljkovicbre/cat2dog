@ECHO OFF

call conda activate cat2dog
echo Calculating eigenfaces...
python pics.py
PAUSE