import os
abc = 123
print(567)
abc = 777
def changeGlobalValue(input):
    global abc
    abc = input
def printABC():
    print(abc)

def getAb():
    yield 1,2

print(os.path.exists("GPT_SoVITS/text"))