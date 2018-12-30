from glob import glob
import os

def recursiveRename(foldername): 
    foldername = foldername + '/' 
    a = glob(foldername+"*.jpg") 
    for idx, name in enumerate(a): 
        os.rename(name, foldername+str(idx)+".jpg")
