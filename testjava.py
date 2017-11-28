from jpype import *

jvmpath = getDefaultJVMPath()
startJVM(jvmpath, "-ea", "-Djava.class.path=.")
TA = JPackage('test').TestApi
jd = TA()
jd.printData('1234')
s = jd.getData('a')
print(s)
shutdownJVM();
