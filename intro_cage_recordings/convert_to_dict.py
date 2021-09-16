import fnmatch, os, sys
import numpy as np
import pickle

sys.path.append('/home/andrew_work/nu/miller_lab_work/cage_data')
import cage_data

dir_path = '/home/andrew_work/nu/miller_lab_work/PopEMG/labeled_cage_data_objects/'
cage_data_filenames = ['20210618_Pop_Cage_001',
                       '20210618_Pop_Cage_003',
                       '20210618_Pop_Cage_004']
seq_len = 30
behaviors = ['sitting_still', 'pg', 'crawling']
class_nums = {'sitting_still': 0, 'pg': 1, 'crawling': 2}
if __name__ == "__main__":
    # Grab the behavior segments for each video
    vid_dict = {key: None for key in cage_data_filenames}
    for fn in cage_data_filenames:
        with open (dir_path + fn + '.pkl', 'rb' ) as fp:
            my_cage_data = pickle.load(fp)
        behavior_segs = my_cage_data.get_all_data_segment(requires_raw_EMG = True,
                                                         requires_spike_timing = True)

        # Split EMG into dictionary 
        bhv = {key: [] for key in my_cage_data.behave_tags['tag']}
        for seg in behavior_segs:
            bhv[seg['label']].append((seg['timeframe'], seg['EMG']))

        # Combine timestamps and EMGs
        bhv_dict = {key: None for key in behaviors} 
        for behavior in behaviors:
            bhv_total = []
            for timestamp, emg in bhv[behavior]:
                tmp = np.concatenate((timestamp.reshape(-1,1), emg), axis=1)

                # Drop frames off end until we can evenly divide into seq_len
                diff = tmp.shape[0] % seq_len
                if diff > 0:
                    tmp = tmp[:-diff]
                bhv_total.append(tmp.reshape(-1, seq_len, 17))

            # Combine all sitting bouts into one big one - time stamp is first d of each elem
            bhv_dict[behavior] = np.concatenate(bhv_total)

        # Add to video dict
        vid_dict[fn] = bhv_dict

    # Save dictionary to file --> used for interactive plot
    with open('bhv_dict.pkl', 'wb') as f:
        pickle.dump(vid_dict, f, pickle.HIGHEST_PROTOCOL)

    # Convert to format needed to train the VAE
    out = []
    for class_name in class_nums.keys():
        tot_class = []
        for vid in vid_dict.keys():
            tot_class.append(vid_dict[vid][class_name])
        tot_class = np.concatenate(tot_class, axis=0)

        # Remove time stamp
        tot_class = tot_class[:,:,1:]

        # Add class number
        class_num = class_nums[class_name]
        class_vec = np.ones((tot_class.shape[0], 1))*class_num
        vecd_seq = tot_class.reshape(tot_class.shape[0], -1)
        tot_class = np.concatenate((class_vec, vecd_seq), axis=1)
        out.append(tot_class)

    out = np.concatenate(out)
    np.savetxt('/home/andrew_work/nu/miller_lab_work/experiments/timeseries-clustering-vae/data/sit_and_crawl_test/sit_and_crawl_test_TRAIN', out, delimiter=',')
np.savetxt('/home/andrew_work/nu/miller_lab_work/experiments/timeseries-clustering-vae/data/sit_and_crawl_test/sit_and_crawl_test_TEST', out, delimiter=',')
