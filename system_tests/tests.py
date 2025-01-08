import os

locals = "data"
#Print whether file exists
print(f"File exists: {os.path.exists(locals)}")
#Print absolute path
print (f"Absolute path: {os.path.abspath(locals)}")
#Print whether it is a directory
print(f"Is a directory: {os.path.isdir(locals)}")
#Print whether it is a file
print(f"Is a file: {os.path.isfile(locals)}")
#Print contents of the directory
print(f"Directory contents: {os.listdir(locals)}")