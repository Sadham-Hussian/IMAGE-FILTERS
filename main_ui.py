# Simple enough, just import everything from tkinter.
from tkinter import *
from tkinter import filedialog, simpledialog

from PIL import Image, ImageTk

from Filter import (
                    RiverdaleFilter, GothamFilter,
                    BlurFilter, RandomFilter, 
                    GrayscaleFilter, MultipleFilter
                    )
from utils import rgb2gray, fourier, plotFourier, gaussian_filter, read_image_from_path, save_img_at_path

from matplotlib import pyplot as plt
import numpy as np

filter_map = {"gotham": GothamFilter,
              "blur": BlurFilter,
              "riverdale": RiverdaleFilter,
              "gray": GrayscaleFilter,
              "multiple": MultipleFilter}


# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        Frame.__init__(self, master)   

        #reference to the master widget, which is the tk window                 
        self.master = master

        #with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget      
        self.master.title("Image Editor")
       
        frame_vertical= Frame(self)
        frame_horizontal= Frame(self)
        frame_vertical.place(x=0, y=0)
        frame_horizontal.place(x=0, y=60)
        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a menu instance
        menu = Menu(self.master)
        self.master.config(menu=menu)

        # create the file object)
        file = Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Exit", command=self.client_exit)

        #added "file" to our menu
        menu.add_cascade(label="File", menu=file)


        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1, ipadx=100)

        # creating a button instance
        selectFileButton = Button(frame_vertical, text="Select File", command = self.open, padx=50)
        # placing the button on my window
        selectFileButton.grid(column=0, row=0)

        # creating a button instance
        gothamFilterButton = Button(frame_vertical,padx=50, text="Gotham", command = lambda:self.applyAndShowFilter("gotham"))
        gothamFilterButton.grid(column=1, row=0)

        # creating a button instance
        blurFilterButton = Button(frame_vertical, padx=50, text="Blur", command = lambda:self.showBlurFilter())
        blurFilterButton.grid(column=2, row=0)

        # creating a button instance
        fourierButton = Button(frame_vertical, padx=50, text="Plot Fourier Transform", command = self.showFourier)
        fourierButton.grid(column=8, row=0)


        # creating a button instance
        riverdaleFilterButton = Button(frame_vertical,padx=50, text="Riverdale", command = lambda:self.applyAndShowFilter("riverdale"))
        riverdaleFilterButton.grid(row=0,column=4)

        # creating a button instance
        blurFilterButton = Button(frame_vertical, padx=50, text="Save edited image...", command = self.saveFile)
        blurFilterButton.grid(column=9, row=0)


        # creating a button instance
        grayscaleFilterButton = Button(frame_vertical, padx=50, text="Grayscale", command = lambda:self.applyAndShowFilter("gray"))
        grayscaleFilterButton.grid(column=6, row=0)

        # creating a button instance
        sepiaFilterButton = Button(frame_vertical, padx=50, text="Sepia", command = lambda:self.applyAndShowFilter("multiple", GrayscaleFilter, GothamFilter))
        sepiaFilterButton.grid(column=7, row=0)


        # The widget to show the original image.
        self.orignal_img_widget = Label(frame_horizontal)
        self.orignal_img_widget.grid(column =10, row=10, padx=15)

        # The widget to show the modified i.e edited image.
        self.edited_img = Label(frame_horizontal)
        self.edited_img.grid(column =12, row=10, padx=15)


    '''
        Present the open file dialog and then open the image and set the variables
    '''
    def open(self):
        self.selected_path = filedialog.askopenfilename()
        print("Image to be opened: ", self.selected_path)
        self.showImg(self.selected_path)

    '''
        Reads the image from the given path and displays the image in the window
    '''
    def showImg(self, path):
        load = Image.open(path)

        self.original_render = ImageTk.PhotoImage(load)
        self.orignal_img_widget.configure(image=self.original_render)
        self.orignal_img_widget["image"] = self.original_render

        self.edited_img["image"] = self.original_render

        self.master.update_idletasks()
        self.img = read_image_from_path(path)
        self.modified = read_image_from_path(path)

    '''
        Shows the edited image on the window
    '''

    def showEditedImg(self, path):
        load = Image.open(path)
        self.render = ImageTk.PhotoImage(load)
        self.edited_img.config(image=self.render)

        self.edited_img["image"] = self.render
        self.master.update_idletasks()

    '''
        Opens the dialog to input the amount of percentage to produce Blur effect on the image
    '''
    def showBlurFilter(self):
        f = BlurFilter()
        a= simpledialog.askfloat("Blur amount", "Blur percentage", minvalue=0.0, maxvalue=100.0)
        self.modified = f.apply_filter(self.modified, a)
        save_img_at_path(self.modified, ".test2222.png")
        self.showEditedImg(".test2222.png")

    '''
        Gets the filter to be applied as input and applies the corresponding filter on the image
    '''
    def applyAndShowFilter(self, filter, *args):
        filter_to_apply = filter_map[filter]
        f = filter_to_apply()
        self.modified = f.apply_filter(self.modified, *args)
        save_img_at_path(self.modified, ".test2222.png")
        self.showEditedImg(".test2222.png")

    '''
        Displays the fourier transform of the modified image
    '''
    def showFourier(self):
        plotFourier(fourier(self.modified))

    '''
        Saves the edited image in a path    
    '''
    def saveFile(self):

        save_path = filedialog.asksaveasfilename()
        if save_path:
            print("Image will be saved at: ", save_path)
            save_img_at_path(self.modified, save_path)

    def client_exit(self):
        exit()


# root window created. Here, that would be the only window
root = Tk()

root.geometry("1200x2000+200+200")

'''
    Attach the Window in to the root
'''
app = Window(root)


#mainloop 
root.mainloop()