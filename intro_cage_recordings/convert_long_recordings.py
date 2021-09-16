import fnmatch, os, sys
#sys.path.append('/media/storage/millerlab/cage_data-master')

import cage_data
import numpy as np
import pickle
import scipy.io


def cagedata2mat(data_path, file_name):
    print('The file %s is going to be loaded'%(data_path + file_name))
    with open ( os.path.join(data_path,file_name), 'rb' ) as fp:
        my_cage_data = pickle.load(fp)

    my_cage_data.pre_processing_summary()
    save_data = dict()

    # get EMG data
    raw_EMGs = my_cage_data.EMG_diff
    save_data['raw_EMGs'] = raw_EMGs
    print('There are %d channels, and each channel has %d time samples'%(len(raw_EMGs), len(raw_EMGs[0])))

    raw_EMG_timeframe = my_cage_data.EMG_timeframe
    save_data['raw_EMG_timestamps'] = raw_EMG_timeframe

    fs_raw_EMG = my_cage_data.EMG_fs
    save_data['EMGFr'] = fs_raw_EMG
    print('The raw EMG signals are sampled %.3f Hz'%(fs_raw_EMG))

    EMG_names = my_cage_data.EMG_names
    save_data['EMG_names'] = EMG_names
    print(EMG_names)

    # get spike data
    spike_timing = my_cage_data.spikes
    save_data['spike_times'] = spike_timing

    # get binned spikes and EMG envelopes
    print(my_cage_data.binned.keys())

    # To get the binned spike counts
    binned_spike_counts = my_cage_data.binned['spikes']
    save_data['spikes_binned'] = binned_spike_counts

    # To get the rectified, filtered and downsampled EMGs
    filtered_EMG = my_cage_data.binned['filtered_EMG']
    save_data['EMG_filtered'] = filtered_EMG

    # To get the time frame of the binned data
    timeframe = my_cage_data.binned['timeframe']
    save_data['binnedFr'] = timeframe

    # Write behaviors to a .annot file
    print('writing annotations to bento file')
    write_annot(my_cage_data.behave_tags, os.path.join(data_path,file_name.replace('.pkl','.annot')))
    print('done!')
    print('--')
    
    del(my_cage_data)
    scipy.io.savemat(os.path.join(data_path,file_name.replace('.pkl','.mat')), mdict={'data': save_data})
    
    
    
def write_annot(behave_tags, file_name):
    f = open (file_name,'w')
    # write the header--------------------
    f.write('Bento annotation file\n')
    f.write('Movie file(s): \n\n')
    f.write('{0} {1}\n'.format('Stimulus name:',''))
    f.write('{0} {1}\n'.format('Annotation start frame:',1))
    f.write('{0} {1}\n'.format('Annotation stop frame:',behave_tags['end_time'][-1]))
    f.write('{0} {1}\n'.format('Annotation framerate:',30))

    f.write('\n{0}\n'.format('List of channels:'))
    channels = ['behavior']
    for item in channels:
        f.write('{0}\n'.format(item))
    f.write('\n');

    f.write('{0}\n'.format('List of annotations:'))
    labels = list(set(behave_tags['tag']))
    #labels = [item.replace(' ','_') for item in labels]
    for item in labels:
        f.write('{0}\n'.format(item))
    f.write('\n')
    
    # now write the contents---------------
    for ch in channels:
        f.write('{0}----------\n'.format(ch))
        for beh in labels:
            f.write('>{0}\n'.format(beh))
            f.write('{0}\t {1}\t {2} \n'.format('Start','Stop','Duration'))
            
            hits = [i for i, x in enumerate(behave_tags['tag']) if x == beh]
            for i in hits:
                f.write('{0}\t{1}\t{2}\n'.format(behave_tags['start_time'][i],behave_tags['end_time'][i],
                                                 behave_tags['end_time'][i] - behave_tags['start_time'][i]))
            f.write('\n')
        f.write('\n')
    
    f.close()

if __name__ == '__main__':
    cagedata2mat('E:/bento_test/20201020004/', '20201020_Pop_Cage_004.pkl')
# if __name__ == '__main__':
#     if(len(sys.argv) == 2):
#         data_path = os.getcwd()
#         source_name = sys.argv[1]
#     elif(len(sys.argv) == 3):
#         data_path = sys.argv[1]
#         source_name = sys.argv[2]
    
#     print('looking for pickles in ' + os.path.join(data_path,source_name))
#     for root, dirs, files in os.walk(os.path.join(data_path,source_name)):
#         for f in files:
#             if '.pkl' in f:
#                 print(os.path.join(root,f))
#                 cagedata2mat(root, f)