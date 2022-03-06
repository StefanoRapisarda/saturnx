def print_info(text,windows=[output_window,output_file],date=True):
    now = tg.my_cdate()
    if date:
        string = '{} - {}\n'.format(now,text)
    else:
        string = '{}\n'.format(text)
    print(string)
    for window in windows:
        if not window is None:
            if isinstance(window,tk.Text):
                output_window.insert(tk.END,string)
            elif isinstance(window,str):
                with open(window,'a') as outfile:
                    outfile.write('\n')

def make_lightcurve(event_files=[],destination=os.getcwd(),
              tres=1.,tseg=16.,low_en=0.5,high_en=10.,
              output_window=None,output_file=None):

    if isinstance(event_files,str): event_files = list(event_files)

    an = os.path.join(destination,'analysis')

    if not output_file is None:
        with open(output_file,'w') as outfile:
            outfile.write('Computing powers\n')

    if not os.path.isdir(an):
        os.mkdir(an)

    if not average is None:
        power_list = tg.PowerList()

    for event_file in event_files:

        if not os.path.isfile(event_file):
            print_info('Event file {} not found, skipping it'.format(event_file))
            continue

        # I need to read the obs ID
        # In case of NICER events
        if 'mpu' in event_file:
            event_file_name = os.path.basename(event_file)
            obs_id = event_file_name[2:12]

        print_info('Obs ID: {}'.format(obs_id))

        lc_list_name = 'lc_list_E{}_{}_T{}_{}.pkl'
        power_name = 'power_E{}_{}_T{}_{}.pkl'
        folder = os.path.join(an,obs_id)

        print_info('Processing event file: {}'.format(event_file))
        print_info('Energy band: {}-{} keV'.format(low_en,high_en),date=False)

        lc_list_file = os.path.join(folder,lc_list_name)
        if os.path.isfile(lc_list_file):
            print_info('lightcurve list file {} already exists'.format(lc_list_file))
        else:
            print_info('Reading event file')
            try:
                events = tg.read_event(event_file)
                print_info('Done!')
            except Exception as e:
                print_info('Could not open event file')
                print_info(e,date=False)
                print_info('\n',date=False)
                continue

            print_info('Reading GTIs from event file')
            try:
                gti = tg.read_gti(event_file)
                print_info('Done!')
            except Exception as e:
                print_info('Could not read GTI')
                print_info(e,date=False)
                print_info('\n',date=False)
                continue

            print_info('Computing lightcurve')
            try:
                lightcurve = tg.Lightcurve.from_event(events,tres=tres,low_en=low_en,high_en=high_en)
                print_info('Done!')
            except Exception as e:
                print_info('Could not compute lightcurve')
                print_info(e,date=False)
                print_info('\n',date=False)
                continue            

            try:
                print_info('Splitting lightcurve according to segments and GTI')
                lcs = lightcurve.split(gti>=tseg).split(tseg)
                print_info('Done!')
            except Exception as e:
                print_info('Could not compute lightcurve list')
                print_info(e,date=False)
                print_info('\n',date=False)
                continue   

            print_info('Saving LightcurveList in the event folder')
            lcs.save(file_name=lc_list_name,fold=folder)

        power_file = os.path.join(folder,power_name)
        if os.path.isfile(power):
            print_info('power file {} already evists'.format(power_file))
        else:
            try:
                print_info('Computing power spectrum list')
                powers = tg.PowerSpectrum.from_lc(lcs)
                print_info('Done!')
            except Exception as e:
                print_info('Could not compute power list')
                print_info(e,date=False)
                print_info('\n',date=False)
                continue               

            try:
                print_info('Computing average power')
                power = powers.average_leahy()
                print_info('Done!')
            except Exception as e:
                print_info('Could not average power')
                print_info(e,date=False)
                print_info('\n',date=False)
                continue             

            print_info('Saving average power')
            power.to_pickle(power_file)
            print_info('Done!')

            if average:
                power_list += [power]

        print_info('\n',date=False)

    if not average is None:
        try:
            print_info('Computing final average')
            power = power_list.average_leahy()
            print_info('Done!')

            print_info('Saving final average in current working directory')
            name = os.path.join(os.getcwd(),average)
            power.to_pickle(name)
            print_info('Done')
        except Exception as e:
            print_info('Could not compute final average')
            print_info(e,date=False)
            print_info('\n',date=False)