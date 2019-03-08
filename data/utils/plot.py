
from data.utils.data_utils import pa16j2d
from data.utils.data_utils import pa17j3d
from data.utils.data_utils import pa20j3d


def get_poselayout(num_joints):
    if num_joints == 16:
        return pa16j2d.color, pa16j2d.cmap, pa16j2d.links
    elif num_joints == 17:
        return pa17j3d.color, pa17j3d.cmap, pa17j3d.links
    elif num_joints == 20:
        return pa20j3d.color, pa20j3d.cmap, pa20j3d.links