@ECHO OFF

echo Setting up conda environment called cat2dog
call conda create --name cat2dog numpy opencv matplotlib pywavelets scikit-learn pillow
call conda activate cat2dog
call pip install PyQt5
call conda deactivate
echo cat2dog environment has been set up
pause