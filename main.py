# import wifi
import math
import random
import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import mean_squared_error
# from scipy.optimize import minimize
#
# # # Simulated access point locations (x, y coordinates)
# # ap_locations = {
# #     "AP1": (0, 0),
# #     "AP2": (10, 0),
# #     "AP3": (0, 10),
# # }
# #
# # # Simulated signal strength measurements (RSSI values) from each AP
# # # Replace these with real measurements from your Wi-Fi hardware
# # signal_strength = {
# #     "AP1": -50,
# #     "AP2": -60,
# #     "AP3": -70,
# # }
# #
# # # Define the objective function for triangulation
# # def objective_function(coords, measurements):
# #     x, y = coords
# #     error = 0
# #     for ap, (ap_x, ap_y) in ap_locations.items():
# #         distance = np.sqrt((x - ap_x) ** 2 + (y - ap_y) ** 2)
# #         estimated_rssi = -30 - 20 * np.log10(distance)  # Path loss model
# #         error += (estimated_rssi - measurements[ap]) ** 2
# #     return error
# #
# # # Initial guess for user's position (you can start with any value)
# # initial_guess = np.array([5, 5])
# #
# # # Perform triangulation using optimization (minimize the objective function)
# # result = minimize(
# #     objective_function,
# #     initial_guess,
# #     args=(signal_strength,),
# #     method="Nelder-Mead",
# # )
# #
# # # Extract the estimated user's position
#     # estimated_position = result.x
# #
# # print("Estimated User Position:", estimated_position)
#
#
# # Known AP coordinates
# AP1_coords = np.array([1, 1])
# AP2_coords = np.array([10, 1])
# AP3_coords = np.array([5, 10])
#
# # Measured signal strengths
# RSSI_AP1 = -50
# RSSI_AP2 = -60
# RSSI_AP3 = -70
#
# # Calculate distances from APs to the user using the signal strength (path loss model)
# distance1 = 10**((RSSI_AP1 - (-30)) / 20)  # Using a simple path loss model (-30 is a reference value)
# distance2 = 10**((RSSI_AP2 - (-30)) / 20)
# distance3 = 10**((RSSI_AP3 - (-30)) / 20)
#
# # Trilateration calculation
# A = 2 * (AP2_coords - AP1_coords)
# B = 2 * (AP3_coords - AP1_coords)
# C = distance1**2 - distance2**2 - np.dot(AP1_coords, AP1_coords) + np.dot(AP2_coords, AP2_coords)
# D = distance1**2 - distance3**2 - np.dot(AP1_coords, AP1_coords) + np.dot(AP3_coords, AP3_coords)
# print(C)
# user_coords = np.linalg.solve(np.array([A, B]), np.array([C, D]))
#
# print("Estimated User Coordinates:", user_coords)

def random_anchor_gen(number_of_anchors):
    anchors = {i: (np.random.uniform(0, 200, size=(1, 3))) for i in range(number_of_anchors)}
    return anchors


def euclidean_distance(original_object, anchors):
    original_distances = {k: np.linalg.norm(v[0] - original_object) for k , v in anchors.items()}

    return original_distances


def generate_original_signal(path_loss_coefficient, original_distance):
    original_signal_strength = {k: -40 - (10 * path_loss_coefficient * math.log(v, 10)) for k,v in
                                original_distance.items()}

    return original_signal_strength


def generate_error_signal_strength(original_signal_strength, standard_dev):
    error_signal_strength = {k: v + np.random.normal(0, standard_dev) for k,v in
                             original_signal_strength.items()}

    return error_signal_strength


def generate_error_distances(path_loss_coefficient, error_signal_strength):
    error_distances = {k: 10 ** ((-40 - v) / (10 * path_loss_coefficient)) for k , v in
                       error_signal_strength.items()}
    return error_distances

def possible_coordinates(anchors, combinations ,error_distances, original_distances):

    all_possible_coordinates = []
    for com in combinations:
        A = np.array([[2 * (anchors[com[0]][0][0] - anchors[com[1]][0][0]),
                       2 * (anchors[com[0]][0][1] - anchors[com[1]][0][1]),
                       2 * (anchors[com[0]][0][2] - anchors[com[1]][0][2])],
                      [2 * (anchors[com[0]][0][0] - anchors[com[2]][0][0]),
                       2 * (anchors[com[0]][0][1] - anchors[com[2]][0][1]),
                       2 * (anchors[com[0]][0][2] - anchors[com[2]][0][2])],
                      [2 * (anchors[com[0]][0][0] - anchors[com[3]][0][0]),
                       2 * (anchors[com[0]][0][1] - anchors[com[3]][0][1]),
                       2 * (anchors[com[0]][0][2] - anchors[com[3]][0][2])]])

        B = np.array([[(error_distances[com[1]]) ** 2 - (error_distances[com[0]]) ** 2 + (
        anchors[com[0]][0][0]) ** 2 - (anchors[com[1]][0][0]) ** 2 + (anchors[com[0]][0][1]) ** 2 - (
                       anchors[com[1]][0][1]) ** 2 + (anchors[com[0]][0][2]) ** 2 - (anchors[com[1]][0][2]) ** 2],
                      [(error_distances[com[2]]) ** 2 - (error_distances[com[0]]) ** 2 + (
                      anchors[com[0]][0][0]) ** 2 - (anchors[com[2]][0][0]) ** 2 + (anchors[com[0]][0][1]) ** 2 - (
                       anchors[com[2]][0][1]) ** 2 + (anchors[com[0]][0][2]) ** 2 - (anchors[com[2]][0][2]) ** 2],
                      [(error_distances[com[3]]) ** 2 - (error_distances[com[0]]) ** 2 + (
                      anchors[com[0]][0][0]) ** 2 - (anchors[com[3]][0][0]) ** 2 + (anchors[com[0]][0][1]) ** 2 - (
                       anchors[com[3]][0][1]) ** 2 + (anchors[com[0]][0][2]) ** 2 - (anchors[com[3]][0][2]) ** 2]
                      ])

        X = np.dot(np.linalg.inv(A), B)

        temp_list = []
        for i in X.tolist():
            temp_list.extend(i)
        all_possible_coordinates.append(temp_list)

    return all_possible_coordinates


def position_estimation(number_of_anchors, original_object, path_loss_coefficient, standard_dev):

    anchors = random_anchor_gen(number_of_anchors)
    original_distances = euclidean_distance(original_object, anchors)
    original_signal_strength = generate_original_signal(path_loss_coefficient, original_distances)
    error_signal_strength = generate_error_signal_strength(original_signal_strength, standard_dev)
    error_distances = generate_error_distances(path_loss_coefficient, error_signal_strength)
    all_combinations = list(combinations(anchors, 4))
    all_possible_coordinates = possible_coordinates(anchors, all_combinations, error_distances, original_distances)

    return all_possible_coordinates, all_combinations

original_object = np.array([30, 40, 50])
number_of_anchors = [5]
path_loss_coefficient = 1.7
standard_dev = 0.5
for i in range(len(number_of_anchors)):
    all_possible_coordinates, all_combinations = position_estimation(number_of_anchors[i], original_object, path_loss_coefficient, standard_dev)
    if number_of_anchors[i] == 5:
        number_of_clusters = [i for i in range(1, len(all_combinations) + 1, 2)]
    else:
        number_of_clusters = [i for i in range(1, len(all_combinations) + 1, 5)]
    for j in range(len(number_of_clusters)):
        kmeans = KMeans(n_clusters=number_of_clusters[j])
        kmeans.fit(all_possible_coordinates)
        counter = Counter(list(kmeans.labels_))
        final_position = kmeans.cluster_centers_[counter.most_common(1)[0][0]]
        MSE = mean_squared_error(original_object, final_position)

        RMSE = math.sqrt(MSE)

