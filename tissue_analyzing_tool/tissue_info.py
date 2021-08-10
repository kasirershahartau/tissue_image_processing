# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 12:53:47 2021

@author: Shahar Kasirer, Anastasia Pergament 

Methods to analyze cells    
"""

import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

class Cell(object):
    """
        The cell class holds the basic information of each cell. It catagorizes
        the inputed information including average size of the cells and the
        location of the centroid.

    """
    def __init__ (self, avrsize, centroid): #defining the average size and centroid location of cells
        self.avrsize = avrsize #average size (number)
        self.centroid = centroid #centroid location (coordinate)
        self.neighbors = set()
        self.number_neighbors = -1
        self.cell_type = None

    def cellinfo (self): #function to easily call cell information.
        print("Cell average size: " + str(self.avrsize) + ", cell centroid: " + str(self.centroid))
        
    def set_avrsize(self, avrsize): #Setter for average size of cell
        self.avrsize = avrsize
    
    def get_avrsize(self): #Getter for average size of cell
        return self.avrsize
        
    def set_centroid(self, centroid): #Setter for centroid of cell
        self.centroid = centroid
        
    def get_centroid(self): #Getter for centroid of cell
        return self.centroid
        
    
class Tissue(object):
    """
         The tissue class holds the cells of a tissue, and organizes information
         according to cell area and centroid location.

    """
  
    def __init__(self, segmentation): 
        self.cells = dict() #opens up array to be filled with the tissue cells' information
        self.segmentation = segmentation #picture
        
      
    def calculate_cellinfo(self):
        """
        Functions to calculate and organize the cell information. Info includes
        area of each cell, and the centroids of the cells. The average centroid 
        location is also found and documented.
        
        """
        for index in range(int(np.max(self.segmentation))): #establishes range from 0 to largest cell size
            area = self.segmentation[self.segmentation == index + 1].size #the area of cells. Add one to eliminate 0s for the area
            if area < 100 or area > 5000:
                continue
            locations = np.argwhere(self.segmentation == index +1) #Establish locations of the cells. Must be with non-zero numbers/
            centroid = locations.mean(axis = 0) #find the mean coordinate of the centroids on each axis
            cell = Cell(area, centroid) 
            self.add_cell(index + 1, cell) #add cells to list for storage
        self.where_neighbors()
        self.count_neighbors()
        
    
    def plot_centroids(self):
        """
        Function to find and label the centroids of the segmented cells. 
        
        """
        
        fig, ax = plt.subplots(1) #makes a subplot of this figure, easier to enclose figures
        plt.imshow(self.segmentation)
        # shapex, shapey = self.segmentation.shape
        for cell in self.cells.values(): #loop for each cell
            centroidy, centroidx = cell.get_centroid() #centroid is coordinate, this splits into seperate values
            circle = Circle((centroidx, centroidy), 8) #plot the circle using centroid coordinates and a radius
            ax.add_patch(circle) #make circles in places of centroids
            
    def get_cell(self, label): #returns cell that is needed
        if label in self.cells:
            return self.cells[label]
        else:
            return None
    
    def add_cell(self, label, cell): #adds cells to dictionary
        self.cells[label] = cell
    
    def where_neighbors(self):#finds all the neighbors in the tissue
        skeleton_pixels = np.argwhere(self.segmentation == 0) #find zeroes of the segmentation image
        nrows, ncolumns = self.segmentation.shape #separate the coordinates of segmentation 
        for pixel_index in range(skeleton_pixels.shape[0]): #loop to repeat the locator for each cell
            pixel_row, pixel_column = skeleton_pixels[pixel_index] #seperates the pixel to identify where the nearest cells are
            neighbors = self.segmentation[max(pixel_row - 1, 0): min(pixel_row + 2, nrows - 1), max(pixel_column - 1, 0): min(pixel_column + 2, ncolumns - 1)]
                #using max/ min to find cells to each size of cell without jumping to other side of image
            unique_neighbors = np.unique(neighbors[neighbors > 0]) #only add neighbors that are unique to list
            for first_index in range(unique_neighbors.size - 1): #loop to add neighbor to list of cell 1
                for second_index in range(first_index + 1, unique_neighbors.size): #loop to add neighbor to second cell (both lists have same neighbors)
                    first_cell = self.get_cell(unique_neighbors[first_index])
                    second_cell = self.get_cell(unique_neighbors[second_index])
                    if first_cell is not None and second_cell is not None: #make sure both lists have each neighbor             
                        first_cell.neighbors.add(unique_neighbors[second_index])
                        second_cell.neighbors.add(unique_neighbors[first_index])
            
    def count_neighbors(self): #count how many neighbors each cell has
        for cell in self.cells.values():
            cell.number_neighbors = len(cell.neighbors) #len does the counting
            
    def get_cell_types(self, hc_marker_image): #differenciate with the cells, based on a frequency level in one of the 3 channels
        for label in self.cells.keys(): #for each cell that we have
            cell = self.cells[label] 
            cell_pixels = hc_marker_image[self.segmentation == label] #the pixels of each cell within the image, based on boundaries in segmentation
            average_cell_brightness = np.mean(cell_pixels) #average brightness of each cell
            if average_cell_brightness > .5: #if above, it is hair cell, below supporting cell
                cell.cell_type = "HC"
            else:
                cell.cell_type = "SC"
                
    def plot_cell_types(self): #used to actually color in the cells, based on their types 
        cell_types_image = np.ones(self.segmentation.shape + (3,)) #creates tupple for the array initialized with 1
        cell_pixels = self.segmentation == 0 #only where pixel are 0
        cell_types_image[np.repeat(cell_pixels[:, :, np.newaxis], 3, axis = 2)] = 0 #repeat for each channel the operations
        for channel_number in range(3): #only the 3rd channel
            channel = cell_types_image[:,:, channel_number] #using only channel 3 to find frequencies
            for label in self.cells.keys(): #loop through each cell
                cell = self.cells[label]
                if cell.cell_type == "HC": #color based on whether cell is HC or SC
                    cell_color = (1, 0, 0)
                elif cell.cell_type == "SC":
                    cell_color = (1, 1, 1)
                channel[self.segmentation == label] = cell_color[channel_number] #overlay colors on the channel
            cell_types_image[:, :, channel_number] = channel #add channel back to original picture
                
        plt.imshow(cell_types_image)
        
    
            
            
            
            