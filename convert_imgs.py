import os
import shutil
import cv2

dir = os.listdir("MainBenchmark/")

for i, file in enumerate(dir):
    print("\r% {:.2f}".format((i+1)/len(dir)),end="")
    if "jpg" in file:
        shutil.copy("MainBenchmark/"+file,"images/"+file)
    else:
        img = cv2.imread("MainBenchmark/" + file)
        img = cv2.resize(img, (640, 480), interpolation = cv2.INTER_AREA)
        cv2.imwrite("images/"+file.replace("png","jpg"), img)