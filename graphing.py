import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = list(range(50,100,5))
performance = [86, 82, 101, 124, 147, 168, 276, 372, 786, 22215]

plt.bar(objects, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.xlabel('Confidence Level')

plt.show()

# Counts [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 82, 101, 124, 147, 168, 276, 372, 786, 22215, 0]
# Incorrect Counts [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56, 55, 57, 61, 54, 61, 46, 56, 72, 125, 0]