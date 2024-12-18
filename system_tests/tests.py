import os

local_path = "austrian_economics.pdf"
#Print whether file exists
print(f"File exists: {os.path.exists(local_path)}")
#Print absolute path
print (f"Absolute path: {os.path.abspath(local_path)}")
