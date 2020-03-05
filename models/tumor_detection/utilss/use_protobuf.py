import os
import sys

args = sys.argv
directory = args[1]
for file in os.listdir(directory):
    if file.endswith(".proto"):
        os.system("protoc" + " " + directory + "/" + file + " --python_out=.")