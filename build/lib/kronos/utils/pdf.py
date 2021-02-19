from fpdf import FPDF

class pdf_page(FPDF):
    
    def __init__(self, page_size = 'A4',margins=[25,25,25,25],*args, **kwargs):
        '''
        PARAMETERS
        ----------
        margins: list
            left,top,right,bottom (<,^,>,v)

        HISTORY
        -------
        2021 01 22, Stefano Rapisarda (Uppsala), creation date
        '''
        super().__init__(*args,**kwargs)
        
        if page_size == 'A4':
            self._width=210
            self._length=297
            self._margins=margins
            
            self._eff_width = self._width - self._margins[0]-self._margins[2]
            self._eff_length = self._length - self._margins[1]-self._margins[3]
            
    def apply_margins(self,coors,input_margins=[0,0,0,0]):
        '''
        Applies margins to a list of coordinates
        
        PARAMETERS
        ----------
        coors: list or int or float
            List of coordinates 
            [x_left,y_up,x_right,y_down]
            If int or float, all the margins are the same
            
        margins: list (optional)
            List of margins
            [left margin,top margin,right margin,bottom margin]
            
        RETURN
        ------
        new_coors: list
            List of coordinates with margins
        '''
        
        if type(input_margins) == float or type(input_margins) == int:
            margins = [input_margins for i in range(4)]
        elif type(input_margins)==list:
            margins = input_margins
        new_coors = [coors[0]+margins[0],coors[1]+margins[1],
                     coors[2]-margins[2],coors[3]-margins[3]]

        return new_coors
    
    def get_grid_coors(self,input_box=[],grid=[1,1,5,5],margins=0,sel='11'):
        '''
        Returns a list of coordinates of a selected box in a grid 
        [x_up,y_left,x_down,y_right]
        
        PARAMETERS
        ----------
        input_box: list (optional)
            Input coordinates of a box. Default is the full page
            
        grid: list (optional)
            List of number of boxes and margins between boxes:
            list[0] = number of boxes on the x axis (default 1)
            list[1] = number of boxes on the y axis (default 1)
            list[2] = space between boxes along x in mm (default 5)
            list[3] = space between boxes along y in mm (default 5)
            
        margins: list (optional)
            List of margins (default 0) to be applied to the selected box
            [left margin,top margin,right margin,bottom margin]
        
        sel: string (optional)
            Two digit string specifying row and colomn of the selected box.
            Default is '11'
            
        RETURNS
        -------
        coor: list
            List of coordinates of upper left and lower right points of a box
            [x_left,y_up,x_right,y_down]
            
        HISTORY
        -------
        2021 01 21, Stefano Rapisarda (Uppsala), creation date
        '''
        
        n_hor_spaces = grid[1]-1
        n_ver_spaces = grid[0]-1
        
        if input_box == []:        
            effective_width = self._width-grid[2]*n_hor_spaces-self._margins[0]-self._margins[1]
            effective_length = self._length-grid[3]*n_ver_spaces-self._margins[2]-self._margins[3]
        else:
            effective_width  = (input_box[3]-input_box[1])-grid[2]*n_hor_spaces
            effective_length = (input_box[2]-input_box[0])-grid[3]*n_ver_spaces
        
        box_width  = (self._eff_width-(n_hor_spaces*grid[2]))/grid[1]
        box_length = (self._eff_length-(n_ver_spaces*grid[3]))/grid[0]
        
        coors = []
        for i in range(grid[0]):
            for j in range(grid[1]):
                if str(i+1)==sel[0] and str(j+1)==sel[1]:
                    coors = [self._margins[0]+j*(box_width+grid[2]),
                             self._margins[1]+i*(box_length+grid[3]),
                             self._margins[0]+j*(box_width+grid[2])+box_width,
                             self._margins[1]+i*(box_length+grid[3])+box_length]
        
        return self.apply_margins(coors,margins)
                      
            
    def box(self,input_coors=[],grid=[1,1,5,5],sel='11',margins=0,lw=0.5):
        '''
        Makes a rectangle
        '''
        coors = self.get_grid_coors(input_box=input_coors,grid=grid,sel=sel,margins=margins)
    
        # Horizontal lines
        self.line(coors[0],coors[1],coors[2],coors[1])
        self.line(coors[0],coors[3],coors[2],coors[3])
        
        # Vertical lines
        self.line(coors[0],coors[1],coors[0],coors[3])
        self.line(coors[2],coors[1],coors[2],coors[3])   
        
    def print_key_items(self,info,title='',fontsize=12,font='Arial',spacing=3,
                 input_coors=[],grid=[1,1,5,5],sel='11',margins=0,
                 conv=0.35):
        '''
        Writes keywords and corresponding items inside the specified box

        PARAMETERS
        ----------
        info: dictionary
            Dictionary of keys and items to print
        
        title: string (optional)
            Title to print on top of keys and items (default is '')

        fontsize: integer (optional)
            Default is 12

        font: string (optional)
            Default is Arial
        
        spacing: float (optional)
            The space between lines is equal the fontsize divided by 
            this number (default is 3)

        input_coors: list (optional)
            Input coordinates

        grid: list (optional)
            List of values:
            - grid[0] = number or rows
            - grid[1] = number of columns
            - grid[2] = space between rows
            - grid[3] = space between columns

        sel: string (optional)
            Two digit string, the first digit is the row index, the 
            second the column index

        margins: float, int, or list
            Margins

        conv: float
            This is the estimated size of the used font in points
            Default is 3.5 (Arial font)
        '''
        
        keys = list(info.keys())
        items = [info[key] for key in keys]
        max_width = max([len(key) for key in keys])
        key_width = (max_width)*conv*fontsize*2/3
        
        # Finding inner box for text (a)
        coors = self.get_grid_coors(input_box=input_coors,grid=grid,sel=sel,margins=margins)
        
        # Printing title
        self.set_xy(coors[0],coors[1])
        input_coors = coors.copy()
        if title != '':
            self.set_font(font,'BU',fontsize+4)
            self.cell(0,conv*(fontsize+fontsize/spacing),txt=title,ln=2,align='L')
            input_coors[1] = coors[1]+conv*(fontsize*2.5)
            
        # Printing keys
        self.print_column(rows=keys,fontexp='BU',xy=(input_coors[0],input_coors[1]))
        
        # Printing items
        input_coors[0] = coors[0]+key_width
        self.print_column(rows=items,fontexp='',xy=(input_coors[0],input_coors[1]))

            
    def print_column(self,rows,title='',fontsize=12,font='Arial',spacing=3,fontexp='',
                 xy=(),conv=0.35):
        '''
        Writes a column of items inside astarting from the specified point
        '''
        
        start_xy = (self.get_x(),self.get_y())
        if len(xy)==0:
            self.set_xy(*start_xy)
        else:
            self.set_xy(*xy)
        
        # Printing title
        if title != '':
            if title == ' ': 
                self.set_font(font,'',fontsize+4)
            else:
                self.set_font(font,'BU',fontsize+4)
            self.cell(0,conv*(fontsize+fontsize/spacing),txt=title,ln=2,align='L')
            self.set_xy(xy[0],xy[1]+conv*(fontsize*2.5))
            
        # Printing rows
        for row in rows:
            self.set_font(font,fontexp,fontsize)
            self.cell(0,conv*(fontsize+fontsize/spacing),txt = row,ln=2,align='L')  
            
        self.set_xy(*start_xy)