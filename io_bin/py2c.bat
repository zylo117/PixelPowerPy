@echo off
REM 判断文件类型
set "str=%~x1"
REM 文件全名为: %~nx1, 文件名为： %~n1, 扩展名为： %~x1
REM 检测变量%str%，即拖拽文件的扩展名，情况不存在则在当前目录打开cmd
if "%str%"==".py" (goto A) else if "%str%"==".pyx" (goto B) else if "%str%"==".c" (goto C) else exit
pause
exit

:A
REM 如果文件是*.py，开始转换
echo Translating to C

REM 编写setup.py
cd.>.\setup.py
echo ^from distutils.core import setup^ >>.\setup.py
echo ^from Cython.Build import cythonize^ >>.\setup.py
echo ^setup(ext_modules = cythonize("%~nx1x"))^ >>.\setup.py

REM py改格式为pyx
rename %~nx1 %~nx1x

REM 获取当前目录名
for /f "delims=" %%i in ("%cd%") do set folder=%%~ni
echo %folder%

REM 开始转换py到c
python setup.py build_ext --inplace

REM 删除多余文件
rd /s /q .\build
del /q %~n1.c

exit

:B
REM 如果文件是*.pyx，不改名，直接开始转换
echo Translating to C

REM 编写setup.py
cd.>.\setup.py
echo ^from distutils.core import setup^ >>.\setup.py
echo ^from Cython.Build import cythonize^ >>.\setup.py
echo ^setup(ext_modules = cythonize("%~nx1"))^ >>.\setup.py

REM 获取当前目录名
for /f "delims=" %%i in ("%cd%") do set folder=%%~ni
echo %folder%

REM 开始转换py到c
python setup.py build_ext --inplace

REM 删除多余文件
rd /s /q .\build
del /q %~n1.c

exit


:C
REM 编写setup.py
cd.>.\setup.py
echo ^from distutils.core import setup^ >>.\setup.py
echo ^from Cython.Build import cythonize^ >>.\setup.py
echo ^setup(ext_modules = cythonize("%~nx1"))^ >>.\setup.py

REM 获取当前目录名
for /f "delims=" %%i in ("%cd%") do set folder=%%~ni
echo %folder%

REM 开始转换py到c
python setup.py build_ext --inplace

REM 删除多余文件
rd /s /q .\build
del /q %~n1.c

exit