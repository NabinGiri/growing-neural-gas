# Implementation of growing neural gas

This is the implemenation of growing neural gas paper (Fritzke, Bernd. "A growing neural gas network learns topologies." Advances in neural information processing systems 7 (1994).). Growing neural gas algorithm unlike any other algorithm does not make predictions, instead, it learns
the topological structure of the data in form of the graph. After training of the network, the structure
of the data is inferred from the graph.

# Files
`growing_neural_gas.py` -> program that implements the growing neural gas \
`org_astro.png` -> original astronaut image used as input to growing neural gas \
`astro.png` -> final output after running growing neural gas on astronaut\
`astro.gif` -> gif of astronaut from growing neural gas \
`org_circle.png` -> original circle image used as input to growing neural gas \
`circle.png` -> final output after runing growing neural gas on circle \
`circle.gif` -> gif of circle from growing neural gas \
`org_moon.png` -> original moon image used as input to growing neural gas \
`moon.png` -> final output after runing growing neural gas on moon \
`moon.gif` -> gif of moon from growing neural gas \
`org_blob.png` -> original blob image used as input to growing neural gas \
`blob.png` -> final output after runing growing neural gas on blob \
`blob.gif` -> gif of blob from growing neural gas 

# Usage
Run `growing_neural_gas` to train the model ; change the which data to use from `DATA = DATA_1 or DATA_2` from line 32 \
Output will be saved for each iteration under the directory `gng/images/*png`. The final `.gif` will be generated using the saved `.png` images. \
The `.gif` file will be saved on the same directory where the program resides








