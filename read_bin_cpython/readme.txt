Cython食用指南：
1.把xxx.py或者xxx.pyx直接拖进去就行了。windows版cython输出的是编译后的.\*模组名*\xxx.pyd，以及对应的py源码pyx
2.然后在Python里面直接import xxx就可以用了

需要：
1.安装cython
2.安装C编译器，不推荐gcc系列，只推荐臃肿的VS buildtool（版本号必须和当前Python的C版本对应）
	目前Python 3.6对应版本是VS 2015（VS 14.0）,亲测用2017是不行的。。。
	但是你又必须安装2017，用它的安装器安装2015，除非你保留了离线安装包，因为2015 buildtool的官方安装包已经被你软删了