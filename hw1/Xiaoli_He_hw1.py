import os
directory = "."
extension = ".png"
files = [file for file in os.listdir(directory) if file.lower().endswith(extension)]

for file in files:
print r"\begin{figure}[!ht]"
print r"\centering"
print r"\includegraphics[width=10cm,height=10cm]{%s}" % file
print r"\caption{File %s}" % file
print r"\label{Serie}"
print r"\end{figure}"

