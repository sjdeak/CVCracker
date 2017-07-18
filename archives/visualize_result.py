from tkinter import *
import pyaux.tk

root = Tk()
root.geometry('300x300')

Label(text=''.join(str(n) for n in lr.result), relief=GROOVE).grid(row=0, columnspan=3, sticky=NSEW)
hand_result = np.array(hr.result, dtype=np.uint8).reshape(3,3)

for i in range(3):
    for j in range(3):
        n = hand_result[i][j]
        Label(text=str(n), relief=GROOVE).grid(row=i+1, column=j, sticky=NSEW)

pyaux.tk.set_grid_stretchable(root)
mainloop()