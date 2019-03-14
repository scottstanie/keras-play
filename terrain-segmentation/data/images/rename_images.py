import glob
import os

curnames = glob.glob(".png")

with open('original_names.txt', 'w') as f:
    for idx, name in enumerate(curnames):
        newname = "image%02d.png" % idx
        f.write("%s %s\n" % (newname, name))
        os.rename(name, newname)
