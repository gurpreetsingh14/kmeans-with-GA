# libraries
import numpy
import random
import matplotlib.pyplot

class generate_data:
    '''
    A class to generate and plot data in the the form of clusters for K-means clustering technique.
    '''
    
    def __init__(self,k):
        '''
        A is a primary function

        Parameters
        ----------
        k = int
            Number of clusters
        '''
        self.num_of_clusters = k

    def data_generation(self,
                        num_of_samples, 
                        X_Cooridinate_Start, 
                        X_Cooridinate_End, 
                        Y_Cooridinate_Start, 
                        Y_Cooridinate_End):
        '''
        This function creates sample coordinates.

        Parameters
        ------------
        num_of_samples = int
                        Number of observations required in a cluster
        X_Cooridinate_Start = int
                        Minimum X-cooridinate of an observation in the cluster
        X_Cooridinate_End = int
                        Maximum X-cooridinate of an observation in the cluster
        Y_Cooridinate_Start = int
                        Minimum Y-cooridinate of an observation in the cluster
        Y_Cooridinate_End = int
                        Maximum Y-cooridinate of an observation in the cluster

        Returns
        -------------
        return : Arraylist
                Two list each for X-coordinates and Y-coordinates of observations in the cluster
        '''
        cluster_X = numpy.random.random(size=(num_of_samples))
        cluster_X = cluster_X * (X_Cooridinate_End - X_Cooridinate_Start) + X_Cooridinate_Start
        cluster_Y = numpy.random.random(size=(num_of_samples))
        cluster_Y = cluster_Y * (Y_Cooridinate_End - Y_Cooridinate_Start) + Y_Cooridinate_Start
        return cluster_X, cluster_Y

    def raw_data(self):
        '''
        This function generates cluster of obervations/data-points based on the input of the user.

        Returns
        -----------
        return : Dict
                A dictionary of cluster.

        Output example
        ---------------
        {'cluster_0': [[<list-of-X-coordinates],[<list-of-Y-coordinates]],
                'cluster_1': [[<list-of-X-coordinates],[<list-of-Y-coordinates]],
                ..
                ..
                'cluster_n': [[<list-of-X-coordinates],[<list-of-Y-coordinates]] }
        '''
        k = self.num_of_clusters
        cluster_map = {}
        input_list = []
        for i in range(k):
            input_list.append([random.randint(25,60),random.randint(0,100),random.randint(20,100),random.randint(0,100),random.randint(20,100)])
        
        for i in range(len(input_list)):
            X_coordinate_list = data_generation(input_list[i][0],input_list[i][1],input_list[i][2],
                                                input_list[i][3],input_list[i][4])[0]
            Y_coordinate_list = data_generation(input_list[i][0],input_list[i][1],input_list[i][2],
                                                input_list[i][3],input_list[i][4])[1]
            cluster_map[f'cluster_{i}'] = [X_coordinate_list,Y_coordinate_list]
        return cluster_map

    
    def plot_input(self,input_data):
        '''
        This function plots the input data on a 2-D plane.

        Parameters
        -------------
        input_data = Dict
                    Data generated from @raw_data() function

        Returns
        -------------
        return: 2-D plot
        '''
        for i in range(len(input_data)):
            coordinate_list = input_data[f'cluster_{i}']
            matplotlib.pyplot.scatter(coordinate_list[0],
                                    coordinate_list[1])

        return matplotlib.pyplot.show()