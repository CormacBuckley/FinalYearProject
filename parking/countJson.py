import json

# some JSON:
x = open("../datasets/parking/train/via_region_data.json", 'r')

data = x.read()
# parse x:
y = json.loads(data)
# the result is a Python dictionary:
numEmpty = 0
numFull = 0
for element in y.values():
    for i in range(0,(len(element["regions"]))):
        if element["regions"][1]["region_attributes"]["Type"] == "Empty":
            numEmpty += 1
        else:
            numFull += 1
print(numEmpty, "Empty spaces", "\n", numFull, "Full Spaces")