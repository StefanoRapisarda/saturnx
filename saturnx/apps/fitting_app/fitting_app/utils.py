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