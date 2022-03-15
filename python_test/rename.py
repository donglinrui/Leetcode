import os,sys
import shutil
#print(os.listdir('./1'))
def copyFiles(sourcefile,targetDir):
    try:
        shutil.copyfile(sourcefile,targetDir)
    except IOError as e:
        print("Unable to copy file. %s" % e)
        exit(1)
    except:
        print("Unexpected error:", sys.exc_info())
        exit(1)

path = './1/'
list = os.listdir(path)
print(list)
os.rename(path + list[0],path + list[0]+'231')
copyFiles(path + list[0]+'231','./'+list[0]+'231')
