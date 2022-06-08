import sys
import numpy as np
sys.path.append('/Volumes/Samsung_T5/saturnx')
from saturnx.utils.pdf import pdf_page

def init_sherpa_model(sherpa_model,name=None,
        parvals=None,frozen=None,mins=None,maxs=None):
    '''
    Function to initialize sherpa model

    PARAMETERS
    ----------
    model: Sherpa model (not an istance)
    ''' 

    if not name is None:
        model = sherpa_model(name)
    else:
        model = sherpa_model()

    for i,par in enumerate(model.pars):
        if not parvals is None: par.val = parvals[i]
        if not frozen  is None: par.frozen = frozen[i]
        if not mins    is None: par.min = mins[i]
        if not maxs    is None: par.max = maxs[i]

    return model

def make_sherpa_result_dict(sherpa_result):
    '''
    Makes a dictionary containing information from fit result
    '''

    result_dict = {}
    lines=sherpa_result.__str__().split('\n')
    for line in lines:
        key = line.split('=')[0].strip()
        item = line.split('=')[1].strip()
        result_dict[key]=item
    return result_dict

def print_fit_results(stat_dict,fit_pars,plot1,plot2,output_name):
    stat_dict1 = {key:stat_dict[key] for key in list(stat_dict.keys())[:8]}
    stat_dict2 = {key:stat_dict[key] for key in list(stat_dict.keys())[8:]}

    # Preparing model parameters arrays
    par_names = []
    par_values = []
    par_perror = []
    par_nerror = []
    for key,item in fit_pars.items():
        for name,val,status,error in zip(item['par_names'],item['par_values'],item['frozen'],item['errors']):
            frozen = ('(frozen)' if status else '(free)')
            par_names += ['{:2}) {:>6}{:>8}'.format(key,frozen,name)]
            par_values += [f'{val:>20.6}']
            if error[0] == np.NaN:
                par_perror += ['+{:>20}'.format('NaN')]
            else:
                par_perror += [f'+{error[0]:>20.6}']
            if error[1] == np.NaN:
                par_nerror += ['-{:>20}'.format('NaN')]            
            else:
                par_nerror += ['-'+f'{error[1]:>20.6}'.replace('-','')]   
    par_info = [par_names,par_values,par_perror,par_nerror]

    pdf = pdf_page(margins=[10,10,10,10])
    pdf.add_page()

    # Printing fit statistics
    pdf.print_key_items(title='Fit statistics',info=stat_dict1,grid=[2,2,5,5],sel='11')
    pdf.print_key_items(title=' ',info=stat_dict2,grid=[4,2,5,5],sel='12')
    
    # Print parameters
    for i in range(1,5):
        coors = pdf.get_grid_coors(grid=[4,4,5,5],sel='2'+str(i))
        if i == 1:
            title = 'Fitting parameters'
        else:
            title = ' '
        pdf.print_column(title=title,rows=par_info[i-1],xy=(coors[0],coors[1]))
        
    # Plotting plots
    pdf.add_page()
    coors = pdf.get_grid_coors(grid=[2,1,5,5],sel='11',margins=[10,0,0,0])
    pdf.image(plot1,x=coors[0],y=coors[1],h=coors[3]-coors[1])
    coors = pdf.get_grid_coors(grid=[2,1,5,5],sel='21')
    pdf.image(plot2,x=coors[0],y=coors[1],h=coors[3]-coors[1])

    pdf.output(output_name,'F')