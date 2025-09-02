CALL python setup.py bdist_wheel

FOR /F "eol=| delims=" %%I IN ('DIR ".\dist\*.whl" /A-D /B /O-D /TW 2^>nul') DO SET "NewestFile=%%I" & GOTO FoundFile
ECHO No *.jpg file found!
GOTO :EOF

:FoundFile
ECHO Newest *.jpg file is: "%NewestFile%"

CALL pip install --upgrade  --force reinstall dist\%NewestFile%
pause