import random
from random import shuffle
filename = "yacht_hydrodynamics.csv"
with open(filename) as file:
    lines = file.readlines()
# shuffle(lines)
# shuffle(lines)
# shuffle(lines)
# shuffle(lines)
# shuffle(lines)

with open('./yacht_hydrodynamics.csv', 'w', encoding='utf-8') as f:
    for line in lines:
        line = line.replace(" ", ",")
        f.write(line)