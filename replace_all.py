# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:57:31 2023

@author: bergeru
"""

import os, fnmatch
def findReplace(directory, find, replace, filePattern):
    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, filePattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            s = s.replace(find, replace)
            with open(filepath, "w") as f:
                f.write(s)
                
findReplace("./", "from viscous_porepy", "from viscous_porepy", "*.py")