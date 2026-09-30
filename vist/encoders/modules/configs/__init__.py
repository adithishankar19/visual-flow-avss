import os.path as osp
import glob
__all__=['SKELETON']

def give_idxs_from_labels(skel):
    skel['label_ids'] = {}
    for i in range(len(skel['keypoint_info'])):
        name = skel['keypoint_info'][i]['name']
        skel['label_ids'][name] = i
    return skel

def get_files_with_extension(directory, extension):
    # Construct the search pattern
    search_pattern = osp.join(directory, f'*.{extension}')
    # Use glob to find files matching the pattern
    files = glob.glob(search_pattern)
    return files[0]

# SKEL_WITH_IDXS = give_idxs_from_labels(SKELETON)
