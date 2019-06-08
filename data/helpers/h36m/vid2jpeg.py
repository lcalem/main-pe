#!/usr/bin/env python2

import os
import sys

import cv2


def makedir(folder_path):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)


def get_mode(subject):
    if subject in [2, 3, 4, 10]: # Test
        return 'testing', 'TEST'

    elif subject in [1, 5, 6, 7, 8]:
        return 'training', 'TRAIN'

    elif subject in [9, 11]:
        return 'training', 'VALID'

    else:
        raise Exception('Unexpected subject number ({})'.format(subject))


def extract_frames(img_folder, mp4_name, prefix):

    subject = int(prefix[5:7])
    folder, mode = get_mode(subject)
    print (prefix, subject)

    mp4_path = os.path.join(folder, 'subject', mp4_name)
    vidcap = cv2.VideoCapture(mp4_path)

    if (prefix[-1] == '') or (prefix[-1] == '\n'):
        prefix = prefix[:-1]

    imgdir = img_folder + '/' + str(prefix)
    makedir(imgdir)

    f = 0
    while True:
        success, image = vidcap.read()
        f += 1
        if not success:
            print ('%s End of file!' % mp4_path)
            break

        # exclude some frames
        if mode == 'TEST' or (mode == 'TRAIN' and (f % 5) != 1) or (mode == 'VALID' and (f % 64) != 1):
            continue
        
        print('%s %d' % (prefix, f))
        fname = imgdir + '/%05d.jpg' % f
        cv2.imwrite(fname, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    vidcap.release()


# python2 vid2jpeg.py /share/DEEPLEARNING/datasets/h36m/vid2jpeg_map.txt /share/DEEPLEARNING/datasets/h36m/images
if __name__ == '__main__':
    try:
        map_path = sys.argv[1]
        output_dir = sys.argv[2]
    except Exception as e:
        print (str(e) + '\n\nExpected the first argument to be the input file, second to be output images folder path')
        sys.exit()

    with open(map_path, 'r') as f_map:
        for line in f_map:
            try:
                mp4_name, prefix = line.split(':')
                extract_frames(output_dir, mp4_name, prefix)
            except Exception as e:
                print (e)

