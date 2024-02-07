from GPT_SoVITS.test2 import *
import os
print("outter")
changeGlobalValue(465)
printABC()

result = []
for item in getAb():
    result.append(item)
print(result[0][1])
