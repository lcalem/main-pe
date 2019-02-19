'''
from the subjects videos of raw h36m dataset to frames
'''

import os
import sys

import cv2


def makedir(folder_path):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)


def extract_frames(img_folder, vidcap, prefix):
    if (prefix[-1] == '') or (prefix[-1] == '\n'):
        prefix = prefix[:-1]

    subject = int(prefix[5:7])
    print (prefix, subject)

    imgdir = img_folder + '/' + str(prefix)
    makedir(imgdir)

    f = 0
    while True:
        success, image = vidcap.read()
        f += 1
        if not success:
            print ('End of file!')
            break

        if subject in [2, 3, 4, 10]: # Test
            continue
        elif subject in [1, 5, 6, 7, 8]: # Train
            if (f % 5) != 1:
                continue
        elif subject in [9, 11]: # Validation
            if (f % 64) != 1:
                continue
        else:
            raise Exception('Unexpected subject number ({})'.format(subject))

        print ('%s %d' % (prefix, f))
        fname = imgdir + '/%05d.jpg' % f
        cv2.imwrite(fname, image, [cv2.IMWRITE_JPEG_QUALITY, 90])


# python2 <PATH>/datasets/h36m/h36m_frames_map.txt <PATH>/datasets/h36m/images
if __name__ == '__main__':
    try:
        map_path = sys.argv[1]
        output_dir = sys.argv[2]
    except Exception as e:
        print (str(e) + '\n\nExpected the first argument to be the input file, second to be output images folder path')
        sys.exit()

    with open(map_path, 'r') as f_map:
        try:
            while True:
                line = f_map.readline()
                if line == '':
                    break
                mp4_name, prefix = line.split(':')
                vidcap = cv2.VideoCapture(mp4_name)
                extract_frames(output_dir, vidcap, prefix)
                vidcap.release()
        except Exception as e:
            print (e)
