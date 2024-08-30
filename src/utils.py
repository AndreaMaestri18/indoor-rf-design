from sionna.rt import Transmitter, Receiver
import tensorflow as tf
import numpy as np

tf.random.set_seed(1)
def generate_signal_map(scene, cm_cell_size, transmitter, preview=False, receiver=None, height=None, orientation=None, size=None, diffraction=False, scattering=False):
    #place TX on top of building (with 8 meters extra)
    try:
        scene.remove("tx")
    except:
        print("no tx")

    try:
        scene.remove("rx")
    except:
        print("no rx")
    tx = Transmitter("tx", position=transmitter["position"],
                    orientation=transmitter["orientation"])

    tx.look_at(transmitter["lookat"])
    scene.add(tx)

    if receiver != None:
        rx = Receiver("rx", position=receiver["position"])
        scene.add(rx)
    
    if height != None:
        cm = scene.coverage_map(num_samples=10e6, # Reduce if your GPU does not have enough memory
                            cm_cell_size=cm_cell_size,
                            diffraction=diffraction, scattering=scattering,  # Enables diffraction and scattering in addition to reflection and LoS
                            check_scene=False,
                            cm_center=(0,0,height),
                            cm_size=size,
                            cm_orientation=orientation
                            ) # Don't check the scene prior to compute to speed things up)
    else: 
        cm = scene.coverage_map(num_samples=10e6, # Reduce if your GPU does not have enough memory
                            cm_cell_size=cm_cell_size,
                            diffraction=diffraction, scattering=scattering,  # Enables diffraction and scattering in addition to reflection and LoS
                            check_scene=False,
                            )
    map = 10.*np.log10(cm.as_tensor()[0].numpy())
    if preview == True:
        scene.preview(coverage_map=cm)
    return map, cm

import scipy.stats
def map_to_probability(signal, map, percentage=1e-1):
    prob_map = np.zeros(np.shape(map))
    if signal != -np.inf:
        signal_meaured = signal+1e-2*np.abs(signal)*np.random.normal()
        nd = scipy.stats.norm(signal_meaured,np.abs(signal_meaured)*percentage)
        for idx,element in enumerate(map):
            for idx2,j in enumerate(element):
                prob_map[idx][idx2] = nd.pdf(j)
    else:
        for idx,element in enumerate(map):
            for idx2,j in enumerate(element):
                if j == -np.inf:
                    prob_map[idx][idx2] = 1
        prob_map /= len(prob_map.flatten())

    return prob_map