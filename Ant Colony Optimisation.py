import xmltodict
import numpy as np
import random
import math

# Global ACO Paramters
ALPHA = 1
BETA = 1
NUM_ANTS = 15
ELITE_ANTS = 15 # Set to same value as NUM_ANTS to remove elitist variation
ITERATIONS = math.floor(10000/NUM_ANTS) # Terminate algorithm when 10000 fitness evaluations are made
RHO = 0.02
Q = 2.5
MAX_PHEROMONE = -1 # Set to -1 to ignore
MIN_PHEROMONE = -1 # Set to -1 to ignore
THETA = 1
FILENAME = 'Burma.xml'
BEST_LENGTH = 999999999
BEST_PATH = []

# Converts the named xml file into a 2D distance matrix
def createDistanceMatrix(fileName):

    # Read the XML file
    with open(fileName) as file:
        data_dict = xmltodict.parse(file.read())    

    # Get vertices and number of cities 
    vertices = data_dict['travellingSalesmanProblemInstance']['graph']['vertex']
    num_cities = len(vertices)

    # Initialize the distance matrix with zeros
    distance_matrix = np.zeros((num_cities, num_cities))

    # Fill the distance matrix with file distances
    city_index = -1
    for vertex in vertices:
        city_index += 1 
        edges = vertex['edge'][0:]

        neighbour_index = -1
        for edge in edges:
            neighbour_index += 1

            if neighbour_index == city_index:
                neighbour_index += 1

            cost = float(edge['@cost'])
            distance_matrix[city_index, neighbour_index] = cost

    return distance_matrix

    # Sum the length of every edge in the path to get total length
def calculatePathLength(path, d):

    total_length = 0

    # Add the individual edges to the cumalititve total
    for i in range(len(path) - 1):
        total_length += d[path[i]][path[i+1]]

    # Add the distance from the last node to the first node as the path is a cycle
    total_length += d[path[-1]][path[0]]
    
    return total_length

# Increase the amount of pheromones in the pheromone matrix for the path taken by the ant
def depositPheromone(path,length, d, T):

    # Delta is the amount of pheromone to deposit, calculated by the fitness function Q / total length of path
    global BEST_LENGTH, BEST_PATH
    if length < BEST_LENGTH:
        BEST_LENGTH = length
        BEST_PATH = path
    delta = Q / length

    # Add the pheromone value to the correct edge in the pheromone matrix
    for i in range(len(path) - 1):
        T[path[i]][path[i+1]] += delta

    # Add pheromone to the edge between the last city and first as the path is a cycle
    T[path[-1]][path[0]] += delta

    return T

# Runs the ant colony optimization algorithm, returns the best route found during runtime
def antColony():

    global ITERATIONS, NUM_ANTS, ALPHA, BETA, RHO, BEST_LENGTH, BEST_PATH, THETA

    # Use xml file to create distance matrix
    d = createDistanceMatrix(FILENAME)
    num_cities = len(d[0])

    
    if MAX_PHEROMONE == -1:
        # Initialise pheromone matrix with random values between 0 and 1
        T = [[ random.random() for j in range(num_cities)] for i in range(num_cities)]
    else:
        # Initialise pheromone matrix to upper bound of pheromone range when using MMAS
        T = [[ MAX_PHEROMONE for j in range(num_cities)] for i in range(num_cities)]

    for iteration in range(ITERATIONS):
        current_paths = [[] for i in range(NUM_ANTS)]

        # Ensure pheromones are within their boundary (if applicable)
        for i in range(num_cities):
            for j in range(num_cities):
                if MAX_PHEROMONE != -1:
                    T[i][j] = min(T[i][j], MAX_PHEROMONE)
                if MIN_PHEROMONE != -1:
                    T[i][j] = max(T[i][j], MIN_PHEROMONE)


            
        for ant in range(NUM_ANTS):
            # All ants start at city 0 - add this to their recorded paths and remove column 0 from distance matrix
            current_paths[ant].append(0)

            # Calculate heuristic matrix
            H = [[1/((d[i][j])**THETA) if (i != j and j != 0) \
                    else 0 for j in range(num_cities)] for i in range(num_cities)]

            for city in range(num_cities - 1):

                # Calculate probabilities of each path
                current_city = current_paths[ant][-1]
                numerator = np.zeros(num_cities)
                row_sum = 0
                for j in range(num_cities):
                    numerator[j] = (T[current_city][j] ** ALPHA) * (H[current_city][j] ** BETA)
                    row_sum += numerator[j]

                probabilities = np.zeros(num_cities)
                for j in range(num_cities):
                    if (numerator[j] == 0 and row_sum == 0):
                        probabilities[j] = 0
                    else:
                        probabilities[j] = numerator[j] / row_sum

                # Choose next city to visit with probabilities
                num = random.random()
                cumalitive_total = 0
                current_city = current_paths[ant][-1]
                next_city = -1

                while cumalitive_total < num:
                    next_city += 1
                    
                    cumalitive_total += float(probabilities[next_city])

                # Move ant to new city         
                current_paths[ant].append(next_city)

                # Remove city column from H
                for i in range(num_cities):
                    H[i][next_city] = 0

        # Update pheromones

        # Evapourate current pheromones
        T = [[ (1-RHO)*T[i][j] for j in range(num_cities)] for i in range(num_cities)]

        paths_dict = []
        for ant in range(NUM_ANTS):
            paths_dict.append({'path':current_paths[ant], 'length': calculatePathLength(current_paths[ant],d)})

        # Sort the paths based on distance
        sorted_paths = sorted(paths_dict, key=lambda x: x['length'])

        # Get top ants
        elite_ants_path = sorted_paths[:ELITE_ANTS]

        # Deposit ant pheromones
        for ant in range(ELITE_ANTS):
            path = elite_ants_path[ant]['path']
            length = elite_ants_path[ant]['length']
            T = depositPheromone(path,length, d, T)

        # Clear paths and repeat iteration with updated pheromone matrix


    # Output the best route found by the ants
    print("Best ant: " + str(BEST_LENGTH))
    # Index cities from 1
    for city in range(len(BEST_PATH)):
        BEST_PATH[city] += 1
    print(BEST_PATH)
    print("\n")
    return BEST_LENGTH


antColony()

        
