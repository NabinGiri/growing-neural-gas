"""
author: nabin
email: nvngiri2@gmail.com
timestamp: Thu Apr 14 2022 5:50 PM
"""
import glob
import os
import os.path as osp
import re
import shutil

import networkx as nx
import numpy as np

import sklearn
from scipy import spatial
from sklearn import datasets
import skimage
from skimage import data, img_as_float, color
from skimage.filters import gaussian, threshold_otsu

import matplotlib.pyplot as plt
import imageio

# which data to use while generating growing neural gas

DATA_0 = sklearn.datasets  # Simple sklearn 2D dataset: examples: moon, circles, blobs
DATA_1 = skimage.data  # data from skimage which contains data from astronaut, camera, horse, bike, scientific images
DATA_2 = None

###############################################
DATA = DATA_1  # make selection HERE!!
###############################################

GNG_ROOT = 'gng'
IMAGES_PATH = GNG_ROOT + '/images/'
if osp.exists(IMAGES_PATH):  # check if the directory preexists, returns True is directory exists
    shutil.rmtree(IMAGES_PATH)  # recursively deletes directory and the files contained in it
os.makedirs(IMAGES_PATH)  # make directory to store the png images which are drawn as growing neural gas iterates

###############################################
# Growing Neural Gas parameters

ITERATIONS = 100  # total number of iterations to run

STEP = 0.1  # error for the unit s1, in paper its e_b

NEIGHBOR_STEP = 0.006  # error for all direct neighbors n of unit s1, this value should be less than e_b, in paper
# its e_n

AFTER_SPLIT_ERROR_DECAY_RATE = 0.5  # used for w_r = after_split_error_decay_rate(w_q + w_f). Weights for the new
# unit r which is in middle of q and f now, in paper its a scalar value 0.5

ERROR_DECAY_RATE = 0.995  # error weight to be applied to all units after iterations

AGE_MAX = 10  # maximum age of the units

STEP_PARAM = 2000


###############################################


class GrowingNeuralGas(object):

    def __init__(self, data):
        self.data = data
        self.graph = nx.Graph()
        self.unit_spawn = 0

    def generate_random_units(self):
        w_a = [np.random.uniform(-2, 2) for _ in
               range(np.shape(self.data)[1])]  # using numpy to randomly generate position
        w_b = [np.random.uniform(-2, 2) for _ in
               range(np.shape(self.data)[1])]
        w_c = [np.random.uniform(-2, 2) for _ in
               range(np.shape(self.data)[1])]
        w_d = [np.random.uniform(-2, 2) for _ in
               range(np.shape(self.data)[1])]
        # using numpy to randomly generate position
        if w_a[0] == w_b[0] and w_a[1] == w_b[1]:  # making sure those two units are not in same location
            self.generate_random_units()  # recursive call to itself
        return np.array(w_a), np.array(w_b), np.array(w_c), np.array(w_d)

    def train_graph(self, step, neighbor_step, age_max, steps_param, after_split_error_decay_rate, error_decay_rate,
                    iter=1):
        """
        :param iter: integer
        :param error_decay_rate: float
        :param after_split_error_decay_rate: float
        :param steps_param: integer
        :param age_max: integer
        :param step: float
        :type neighbor_step: float
        """

        # Step 0 : Start with two units a and b at random positions w_a and w_b
        # program creates 4 initial units, more initial units means the program is faster
        w_a, w_b, w_c, w_d = self.generate_random_units()
        self.graph.add_node(self.unit_spawn, weight=w_a.reshape(1, -1), error=0.0)  # creating new initial unit w_a
        self.unit_spawn += 1
        self.graph.add_node(self.unit_spawn, weight=w_b.reshape(1, -1), error=0.0)  # creating another initial unit w_b
        self.unit_spawn += 1
        self.graph.add_node(self.unit_spawn, weight=w_c.reshape(1, -1), error=0.0)  # creating another initial unit w_b
        self.unit_spawn += 1
        self.graph.add_node(self.unit_spawn, weight=w_d.reshape(1, -1), error=0.0)  # creating another initial unit w_b
        self.unit_spawn += 1

        # Step 1: Iterate through data
        cycle = 0
        for iter_step in range(iter):
            print(f"Iteration:{iter_step + 1}")
            np.random.shuffle(self.data)
            steps_counter = 0
            for input_signal in self.data:
                # Step 2 : Find the nearest unit s_1 and the second-nearest unit s_2, using distance formula

                nearest_units = self.find_nearest_units(input_signal)
                s_1 = nearest_units[0]
                s_2 = nearest_units[1]
                # Step 6: If s_1 and s_2 are connected by an edge, set the age of this edge to zero, if edge does not
                # exist then create it

                # Step 3: Increment ages of all edges emanating from s_1
                for fro, to, attr in self.graph.edges(data=True, nbunch=[s_1]):
                    self.graph.add_edge(fro, to, age=attr['age'] + 1)
                try:
                    # self.graph.nodes[s_1]['error'] += distance[s_1] ** 2
                    self.graph.nodes[s_1]['error'] += spatial.distance.euclidean(input_signal,
                                                                                 self.graph.nodes[s_1][
                                                                                     'weight']) ** 2
                except KeyError:
                    print(self.graph.nodes())
                    print(s_1)
                    print(self.graph.nodes[s_1])

                update_w_s_1 = step * (np.subtract(input_signal, self.graph.nodes[s_1]['weight']))
                self.graph.nodes[s_1]['weight'] = np.add(self.graph.nodes[s_1]['weight'], update_w_s_1)

                for neighbor in self.graph.neighbors(s_1):
                    update_w_n = neighbor_step * (np.subtract(input_signal, self.graph.nodes[neighbor]['weight']))
                    self.graph.nodes[neighbor]['weight'] = np.add(
                        self.graph.nodes[neighbor]['weight'], update_w_n)

                self.graph.add_edge(s_1, s_2, age=0)

                # Step 7: Remove the edges with age larger than age_max, if this has no edges then remove them as
                # well

                self.remove_connections(age_max)

                # Step 4: Add the squared distance between the input signal and the nearest unit in input space to
                # local counter variable

                # Step 5:Move s_1 and its direct topological neighbors towards input_signal by fraction of step and
                # neighbor_step, if the total distance


                # Step 8: If the number of input signals generated so far is an integer multiple of a parameter
                # step_parameter, we add a new unit
                steps_counter += 1
                if steps_counter % steps_param == 0:
                    # save the plots with a unique cycle number
                    self.plot_graph(IMAGES_PATH + str(cycle) + '.png')
                    cycle += 1  # increment the cycle number
                    # Step 8.A: Determine unit q with maximum accumulated error
                    maximum_accum_error_unit = max(self.graph.nodes[unt]['error'] for unt in self.graph.nodes)
                    for key, val in self.graph.nodes.items():
                        if val['error'] == maximum_accum_error_unit:
                            maximum_accum_error_unit = key
                    # Step 8.B : Insert a new unit r halfway between q and its neighbor f with largest error
                    maximum_accum_error_unit_neighbor = max(self.graph.neighbors(maximum_accum_error_unit))
                    for key, val in self.graph.nodes.items():
                        if val['error'] == maximum_accum_error_unit_neighbor:
                            maximum_accum_error_unit_neighbor = key

                    new_unit_weight = 0.5 * (
                        np.add(self.graph.nodes[maximum_accum_error_unit]['weight'],
                               self.graph.nodes[maximum_accum_error_unit_neighbor]['weight']))
                    new_unit = self.unit_spawn
                    self.unit_spawn += 1

                    # Step 8.C : Insert edge connecting the new unit r with units q and f , and remove the original edge
                    # between q and f
                    self.graph.add_node(new_unit, weight=new_unit_weight, error=0.0)
                    self.graph.add_edge(new_unit, maximum_accum_error_unit, age=0)
                    self.graph.add_edge(new_unit, maximum_accum_error_unit_neighbor, age=0)
                    self.graph.remove_edge(maximum_accum_error_unit, maximum_accum_error_unit_neighbor)

                    # Step 8.D : Decrease the error variables of q and f by multiplying them with a constant. Initialize
                    # the error variable of r with the new value of the error variable of q
                    self.graph.nodes[maximum_accum_error_unit]['error'] *= after_split_error_decay_rate
                    self.graph.nodes[maximum_accum_error_unit_neighbor]['error'] *= after_split_error_decay_rate
                    self.graph.nodes[new_unit]['error'] = self.graph.nodes[maximum_accum_error_unit]['error']

                    # Step 8.D : Decrease all error variable by multiplying with constant d
                for unt in self.graph.nodes():
                    try:
                        self.graph.nodes[unt]['error'] *= error_decay_rate
                    except KeyError:
                        print(unt)
                        print(self.graph.nodes[unt])

                    if self.graph.degree(nbunch=[unt]) == 0:
                        print(unt)

        self.plot_graph(IMAGES_PATH + str(cycle) + '.png')

        print(f"Total graph nodes are: {len(self.graph.nodes())})

    def find_nearest_units(self, observation):
        distance = []
        for u, attributes in self.graph.nodes(data=True):
            vector = attributes['weight']
            dist = spatial.distance.euclidean(vector, observation)
            distance.append((u, dist))
        distance.sort(key=lambda x: x[1])
        ranking = [u for u, dist in distance]
        return ranking

    def remove_connections(self, age_max):
        candidate_to_remove = list()
        for u, v, attr in self.graph.edges(data=True):
            if attr['age'] > age_max:
                candidate_to_remove.append((u, v))
        for u, v in candidate_to_remove:
            self.graph.remove_edge(u, v)
        candidate_to_remove = list()
        for u in self.graph.nodes():
            if self.graph.degree(u) == 0:
                candidate_to_remove.append(u)
        for u in candidate_to_remove:
            self.graph.remove_node(u)

    def plot_graph(self, save_path):
        plt.clf()
        node_pos = {}
        for u in self.graph.nodes():
            weight = self.graph.nodes[u]['weight'][0]
            node_pos[u] = (weight[0], weight[1])
        nx.draw(self.graph, pos=node_pos, node_size=10, node_shape="^", style='dashed', node_color='k',
                edge_color='k')
        plt.draw()
        plt.savefig(save_path)


def make_data_0():
    data = None  # initialize data is None at first
    num_samples = 5000
    data_type = 'circles'  # options to choose from blobs, moons and circles:- just replace 'circles' with either 'blobs' or 'moons'
    print(f"Preparing sklearn data: {data_type}")
    if data_type == 'circles':
        data = datasets.make_circles(n_samples=num_samples, noise=0.05, factor=0.5)
    elif data_type == 'blobs':
        data = datasets.make_blobs(n_samples=num_samples, random_state=10)
    elif data_type == 'moons':
        data = datasets.make_moons(n_samples=num_samples, noise=0.05)
    if not isinstance(data,
                      type(
                          None)):  # checking if the data is None or has values in it, if the data is None then terminate
        data = sklearn.preprocessing.StandardScaler().fit_transform(data[0])
    else:
        print(f"The data has not been initialized for preprocessing, got {data}")
        print("Rerun the program and select correct data")
        exit()
    plt.scatter(*data.T, c='k')
    plt.draw()
    # plt.savefig("moons_org.png")

    return data


def make_data_1():
    print(f"Preparing skimage data")
    data = list()
    data_type = img_as_float(skimage.data.astronaut())  # convert the image to floating point format and save
    data_crop = data_type[30:180, 150:300]  # taking the face of astronaut image
    data_grey = color.rgb2gray(data_crop)
    data_blured = gaussian(data_grey, sigma=0.6)
    threshold = threshold_otsu(data_blured) + 0.1
    data_binary = data_blured < threshold
    for (x, y), val in np.ndenumerate(data_binary):
        if val == 1:
            data.append([y, -x])
    data = np.array(data)  # converting list to np array
    plt.scatter(*data.T, c='k')  # plot the data in black color, use 'r' for red, 'b' for blue
    plt.draw()
    # plt.savefig("org_astro.png")
    return data


def convert_images_to_gif(output_images_dir, output_gif):
    """Convert a list of images to a gif."""

    image_dir = "{0}/*.png".format(output_images_dir)
    list_images = glob.glob(image_dir)
    file_names = sort_images(list_images)
    images = [imageio.imread(fn) for fn in file_names]
    imageio.mimsave(output_gif, images)


def sort_images(limages):

    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]

    limages = sorted(limages, key=alphanum_key)
    return limages


if __name__ == "__main__":
    if DATA == DATA_0:
        data = make_data_0()
    elif DATA == DATA_1:
        data = make_data_1()
    print(f"Data processing done.")
    print("Initiating Training of Growing Neural Gas.")
    print("Note: Growing Neural Gas unlike other neural network algorithm does not make predictions, it learns "
          "topological structure of data in the form of graph.")

    growing_neural_gas = GrowingNeuralGas(data)
    growing_neural_gas.train_graph(step=STEP, neighbor_step=NEIGHBOR_STEP, age_max=AGE_MAX, steps_param=STEP_PARAM,
                                   after_split_error_decay_rate=AFTER_SPLIT_ERROR_DECAY_RATE,
                                   error_decay_rate=ERROR_DECAY_RATE, iter=ITERATIONS)
    convert_images_to_gif(IMAGES_PATH, 'gng.gif')
