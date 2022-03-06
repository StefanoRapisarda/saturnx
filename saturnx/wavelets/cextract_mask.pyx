cimport numpy as np
import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.path as mpltPath

def extract_mask_cython(patch,int nf,int nt, str opt='wrf'):
    '''
    Creates a mask of zeros with the same dimension of the wavelet 
    transform object (nf,nt) where only the points inside the 
    specified patch will be one.

    PARAMETERS
    ----------
    patch: shapely.geometry.Polygon
        Significance patch/contour. Only the points inside this patch
        will be 1 in the mask
    nf: int
        Number of wavelet frequency bins
    nt: int
        Number of wavelet time bins
    opt: str
        Option to be used to check if a mask point is inside the given
        patch:
        - wrf: it will use the W. Randolph Franklin point-in-polygon 
        algorithm. This will exclude points on the edge
        - mathlab: it will convert the patch in a matlab path and then
        use the path method contains_point()
        - shapely: it will use the shapely method .within()
        The options are listed in order of decreasing speed. mathlab 
        and shapely also will include points on the edge of the path

    RETURNS
    -------
    mask: nd.array
        (nf,nt) array of integer. A mask ready to be applied to the 
        wavelet transform

    HISTORY
    -------
    2021 07 27, Stefano Rapisarda, Uppsala (creation data)
        This is result of a couple of weeks of research on how to speed
        up python code. The option wrf is significantly faster than my
        original code. As wrf does not account for points on the edge
        of the patch, the masks obtained with this and the other 
        methods differ of some points.
    '''

    cdef double[:] vert_x = np.array(patch.exterior.xy[0])
    cdef double[:] vert_y = np.array(patch.exterior.xy[1])    
    cdef int[:,:] mask = np.zeros((nf,nt),dtype='int32')
    cdef int xi,yi,i,j

    if opt == 'matlab':
        path = mpltPath.Path([[x,y] for x,y in zip(patch.exterior.xy[0],patch.exterior.xy[1])])

    for yi in range(int(patch.bounds[1]),int(patch.bounds[3])+1+1):
        for xi in range(int(patch.bounds[0]),int(patch.bounds[2])+1+1):
            
            inside = False
            if opt == 'wrf':
                j = len(vert_x)-1
                for i in range(len(vert_x)):
                    if ( ((vert_y[i]>yi) != (vert_y[j]>yi)) and \
                        (xi < (vert_x[j]-vert_x[i]) * (yi-vert_y[i]) / (vert_y[j]-vert_y[i]) + vert_x[i]) ):
                        inside = not inside
                    j = i
            elif opt == 'matlab':
                inside = path.contains_point((xi,yi))
            elif opt == 'shapely':
                inside = Point(xi,yi).within(patch)
            else:
                print('Invalid option for extract mask')
                return

                
            if inside: mask[yi,xi] = 1
                           
    return mask

