import os

file = open("/home/jjvyas1/5Dec_segmentation.out")
iou = []
for line in file.readlines():
    if 'IoU' in line and 'Testing' not in line:
        for i in range(len(line)):
            if line[i] == "U":
                ch = i + 3
        iou.append(float(line[ch:]))
print(iou, len(iou))
file.close()
