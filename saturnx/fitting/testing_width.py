import tkinter as tk
from tkinter import ttk

win = tk.Tk()
for i in range(1,6):
    factor=1000
    frame = tk.Frame(win)
    frame.grid(column=0,row=0,sticky='nswe')
    frame1 = ttk.Labelframe(frame,text='frame1',width=i*100,height=i*100)
    frame1.grid(column=0,row=0,padx=5,pady=5)
    frame2 = ttk.Labelframe(frame,text='frame2',width=i*200,height=i*200)
    frame2.grid(column=0,row=1,padx=5,pady=5)
    stop = input('press key')

win.mainloop()