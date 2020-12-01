import tkinter as tk
from tkinter import ttk

window = tk.Tk()
frame = tk.Frame(window)
frame.pack()

text = tk.Text(frame)
text.grid(column=0,row=0)

label = tk.Label(frame,text='This is a label ooooooooooooo')
label.grid(column=0,row=1)

message = tk.Message(frame,text='This is a message ooooooooooo')
message.grid(column=0,row=2)

window.mainloop()