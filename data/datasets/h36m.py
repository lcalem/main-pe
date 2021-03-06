import os

import numpy as np
import scipy.io as sio
from PIL import Image

from data.utils import bbox
from data.utils import transform
from data.utils.camera import Camera
from data.utils.data_utils import TEST_MODE, TRAIN_MODE, VALID_MODE, get_visible_joints

from model.utils import pose_format



def load_h36m_mat_annotation(filename):
    mat = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)

    # Respect the order of TEST (0), TRAIN (1), and VALID (2)
    sequences = [mat['sequences_te'], mat['sequences_tr'], mat['sequences_val']]
    action_labels = mat['action_labels']
    joint_labels = mat['joint_labels']

    return sequences, action_labels, joint_labels


def serialize_index_sequences(seq):
    frames_idx = []
    for s in range(len(seq)):
        for f in range(len(seq[s].frames)):
            frames_idx.append((s, f))

    return frames_idx


def get_clip_frame_index(
    sequence_size, 
    subsample, 
    num_frames,
    random_clip=False):

    # Assert that subsample is integer and positive
    assert (type(subsample) == int) and subsample > 0

    idx_coef = 1.
    while idx_coef * sequence_size < num_frames:
        idx_coef *= 1.5
    sequence_size *= idx_coef

    # Check if the given subsample value is feasible, otherwise, reduce
    # it to the maximum acceptable value.
    max_subsample = int(sequence_size / num_frames)
    if subsample > max_subsample:
        subsample = max_subsample

    vidminf = subsample * (num_frames - 1) + 1 # Video min num of frames
    maxs = sequence_size - vidminf # Maximum start
    if random_clip:
        start = np.random.randint(maxs + 1)
    else:
        start = int(maxs / 2)

    frames = list(range(start, start + vidminf, subsample))
    if idx_coef > 1:
        for i in range(len(frames)):
            frames[i] = int(frames[i] / idx_coef)

    return frames


class Human36M(object):
    '''
    Human3.6M dataset for 3D pose estimation dataset manager
    '''

    def __init__(self, 
                 dataset_path, 
                 dataconf, 
                 poselayout=pose_format.pa17j3d,
                 topology='sequences', 
                 clip_size=16):

        assert topology in ['sequences', 'frames'], 'Invalid topology ({})'.format(topology)

        self.dataset_path = dataset_path
        self.dataconf = dataconf
        self.poselayout = poselayout
        self.topology = topology
        self.clip_size = clip_size
        self.load_annotations(os.path.join(dataset_path, 'annotations.mat'))

    def load_annotations(self, filename):
        try:
            # sequences here is a list of 3: test sequences, train sequences, val sequences
            self.sequences, self.action_labels, self.joint_labels = load_h36m_mat_annotation(filename)
            self.frame_idx = [serialize_index_sequences(self.sequences[0]),
                    serialize_index_sequences(self.sequences[1]),
                    serialize_index_sequences(self.sequences[2])]

            global ACTION_LABELS
            ACTION_LABELS = self.action_labels

        except:
            warning('Error loading Human3.6M dataset!')
            raise


    def get_data(self, key, mode, frame_list=None, fast_crop=False):
        '''
        pose_w:     ground truth data in millimeters
                    loaded from the annotation file and formatted to the self.poselayout format (= picking out joints)
        pose_uvd:   same but with xy coords given in pixels (uv) and depth (d) in millimeters
        pose:       same format as pose_uvd but with all transformations (crop, hflip, etc) applied
        '''
        output = {}

        if mode == TRAIN_MODE:
            dconf = self.dataconf.random_data_generator()
            random_clip = True
        else:
            dconf = self.dataconf.get_fixed_config()
            random_clip = False

        if self.topology == 'sequences':
            seq = self.sequences[mode][key]
            if frame_list is None:
                frame_list = get_clip_frame_index(len(seq.frames),
                        dconf['subspl'], self.clip_size,
                        random_clip=random_clip)
            objframes = seq.frames[frame_list]
        else:
            seq_idx, frame_idx = self.frame_idx[mode][key]
            seq = self.sequences[mode][seq_idx]
            objframes = seq.frames[[frame_idx]]

        # Build a Camera object
        cpar = seq.camera_parameters
        cam = Camera(cpar.R, cpar.T, cpar.f, cpar.c, cpar.p, cpar.k)

        # Load and project the poses
        pose_w = self.load_pose_annot(objframes)
        pose_uvd = cam.project(np.reshape(pose_w, (-1, 3)))
        pose_uvd = np.reshape(pose_uvd, (len(objframes), self.poselayout.num_joints, 3))

        # Compute GT bouding box
        imgsize = (objframes[0].w, objframes[0].h)
        objpos, winsize, zrange = bbox.get_crop_params(pose_uvd[:, 0, :],
                imgsize, cam.f, dconf['scale'])

        objpos += dconf['scale'] * np.array([dconf['transx'], dconf['transy']])
        frames = np.empty((len(objframes),) + self.dataconf.input_shape)
        pose = np.empty((len(objframes), self.poselayout.num_joints, self.poselayout.dim))

        for i in range(len(objframes)):
            image = 'images/%s/%05d.jpg' % (seq.name, objframes[i].f)
            imgt = transform.ImageTransform(Image.open(os.path.join(self.dataset_path, image)))

            imgt.rotate_crop(dconf['angle'], objpos, winsize)
            if dconf['hflip'] == 1:
                imgt.horizontal_flip()

            imgt.resize(self.dataconf.crop_resolution)
            imgt.normalize_affinemap()
            frames[i, :, :, :] = transform.normalize_channels(imgt.asarray(),
                    channel_power=dconf['chpower'])

            pose[i, :, 0:2] = transform.transform_2d_points(imgt.afmat, pose_uvd[i, :,0:2], transpose=True)
            pose[i, :, 2] = (pose_uvd[i, :, 2] - zrange[0]) / (zrange[1] - zrange[0])

            if imgt.hflip:
                pose[i, :, :] = pose[i, self.poselayout.map_hflip, :]

        # Set outsider body joints to invalid (-1e9) 
        pose = np.reshape(pose, (-1, self.poselayout.dim))
        pose[np.isnan(pose)] = -1e9
        v = np.expand_dims(get_visible_joints(pose[:,0:2]), axis=-1)
        pose[(v==0)[:,0],:] = -1e9
        pose = np.reshape(pose, (len(objframes), self.poselayout.num_joints, self.poselayout.dim))
        v = np.reshape(v, (len(objframes), self.poselayout.num_joints, 1))

        pose = np.concatenate((pose, v), axis=-1)
        if self.topology != 'sequences':
            pose_w = np.squeeze(pose_w, axis=0)
            pose_uvd = np.squeeze(pose_uvd, axis=0)
            pose = np.squeeze(pose, axis=0)
            frames = np.squeeze(frames, axis=0)

        output['camera'] = cam.serialize()
        output['action'] = int(seq.name[1:3]) - 1
        output['pose_w'] = pose_w
        output['rootz'] = pose_uvd[0,2]
        output['pose_uvd'] = pose_uvd
        output['pose'] = pose
        output['frame'] = frames 

        # Take the last transformation matrix, it should be the same for all frames
        output['afmat'] = imgt.afmat.copy()

        return output

    def load_pose_annot(self, frames):
        p = np.empty((len(frames), self.poselayout.num_joints, self.poselayout.dim))
        for i in range(len(frames)):
            p[i,:] = frames[i].pose3d.T[self.poselayout.map_from_h36m, 0:self.poselayout.dim].copy()

        return p

    def clip_length(self):
        if self.topology == 'sequences':
            return self.clip_size
        else:
            return None

    def clip_shape(self):
        if self.topology == 'sequences':
            return (self.clip_size,)
        else:
            return ()

    def get_shape(self, dictkey):
        if dictkey == 'frame':
            return self.clip_shape() + self.dataconf.input_shape
        if dictkey == 'pose':
            return self.clip_shape() + (self.poselayout.num_joints, self.poselayout.dim+1)
        if dictkey == 'pose_w':
            return self.clip_shape() + (self.poselayout.num_joints, self.poselayout.dim)
        if dictkey == 'pose_uvd':
            return self.clip_shape() + (self.poselayout.num_joints, self.poselayout.dim)
        if dictkey == 'action':
            return (1,)
        if dictkey == 'camera':
            return (21,)
        if dictkey == 'afmat':
            return (3, 3)
        raise Exception('Invalid dictkey %s on get_shape!' % dictkey)

    def get_length(self, mode):
        if self.topology == 'sequences':
            return len(self.sequences[mode])
        else:
            return len(self.frame_idx[mode])

