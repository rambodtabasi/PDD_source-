#! /usr/bin/env python
# -*- coding: utf-8 -*-
import math
import sys

import numpy as np
import numpy.ma as ma
import scipy.spatial
import matplotlib.path as path
import matplotlib.pyplot as plt

from ensight import Ensight
#from tempfile import TemporaryFile

from PyTrilinos import Epetra
from PyTrilinos import EpetraExt
from PyTrilinos import Teuchos
from PyTrilinos import Isorropia
from PyTrilinos import NOX

#import matplotlib.pyplot as plt
import pylab
import time as ttt


#np.set_printoptions(threshold=np.nan)
## class ##
class PD(NOX.Epetra.Interface.Required,
	NOX.Epetra.Interface.Jacobian):
    """
       Class that inherits from `NOX.Epetra.Interface.Required
       <http://trilinos.sandia.gov/packages/pytrilinos/development/NOX.html>`_
       to produce the problem interface to NOX for solving steady-state
       peridynamic problems.
    """
    def __init__(self, num_nodes, length, width=10.0, bc_regions=None,
            bc_values=None, symm_bcs=False, horizon=None, verbose=None):
        """Instantiate the problem object"""
        NOX.Epetra.Interface.Required.__init__(self)
        NOX.Epetra.Interface.Jacobian.__init__(self)

        #Epetra communicator attributes
        self.feshar_const = 1e-20
        self.comm = Epetra.PyComm()
        self.rank = self.comm.MyPID()
        self.size = self.comm.NumProc()
        self.nodes_numb = num_nodes
        self.width = width
        self.length = length
        self.visc = 1.79e-6

	#Print version statement

        if self.rank == 0: print("PDD.py version 0.4.0zzz\n")

	# Domain properties
        #self.iteration = 0
        self.num_nodes = num_nodes
        self.grid_spacing = float(length) / (num_nodes - 1)
        self.delta_x = self.grid_spacing
        self.delta_y = self.grid_spacing
        self.delta_t = 1e-8
        self.rho  = 1000
        self.bc_values = bc_values
        self.symm_bcs = symm_bcs

        #self.aspect_ratio = 45.0 / num_nodes
        self.aspect_ratio = 1.0
        width = length * self.aspect_ratio
        #width = 0.0
        self.width = width

        #Default settings and attributes
        if horizon != None:
            self.horizon = horizon
        else:
            self.horizon =1.1 * self.grid_spacing

        if verbose != None:
            self.verbose = True
        else:
            self.verbose = False

        #Flow properties

        #Setup problem grid
        self.create_grid(length, width, 0.0 )
        #Find the global family array
        self.get_neighborhoods(width,length)
        #Initialize the neighborhood graph
        #check to see how the neighbors match
        self.__init_neighborhood_graph()
        #Load balance
        self.__load_balance()
        #Initialize jacobian
        self.__init_jacobian()
	#self.__init_overlap_import_export()
        self.__init_overlap_import_export()
        #Initialize grid data structures
        self.__init_grid_data()
    def isinteger(x):
        return np.equal(np.mod(x, 1), 0)
    def create_grid(self, length, width,flag):
        """Private member function that creates initial rectangular grid"""
            #Create grid, if width == 0, then create a 1d line of nodes
        if self.rank ==0 :
            j = np.complex(0,1)
            if width > 0.0:
                grid = np.mgrid[0:length:self.num_nodes*j,
                        0:width:self.num_nodes*self.aspect_ratio*j]
                self.nodes  = np.asarray(zip(grid[0].ravel(),grid[1].ravel()),
                        dtype=np.double)

            else:
                x = np.r_[0.0:length:self.num_nodes*j]
                y = np.r_[[0.0] * self.num_nodes]
                self.nodes = np.asarray(zip(x, y),dtype=np.double)

            my_num_nodes = len(self.nodes)

        else:
            self.nodes = np.array([],dtype=np.double)
            my_num_nodes = len(self.nodes)

        self.__global_number_of_nodes = self.comm.SumAll(my_num_nodes)

        return

    def get_neighborhoods(self,width,length):
	""" cKDTree implemented for neighbor search """
        if self.rank == 0:
            #Create a kdtree to do nearest neighbor search
            tree = scipy.spatial.cKDTree(self.nodes)
            #Get all neighborhoods
            for i in range(len(self.nodes)):
                nodes = self.nodes[i]
                if nodes[1] < self.horizon:
                    self.nodes[i][1]= nodes[1] + width + self.grid_spacing
            self.neighborhoods_down = tree.query_ball_point(self.nodes,
                    r=self.horizon, eps=0.0, p=2)
        self.create_grid(length, width,0)
        if self.rank == 0:
            tree = scipy.spatial.cKDTree(self.nodes)
            for i in range(len(self.nodes)):
                nodes = self.nodes[i]
                if nodes[1] > (width - self.horizon):
                    self.nodes[i][1]= nodes[1] - width - self.grid_spacing
            self.neighborhoods_up = tree.query_ball_point(self.nodes,
                    r=self.horizon, eps=0.0, p=2)
        self.create_grid(length,width,0)
        if self.rank ==0:
            tree = scipy.spatial.cKDTree(self.nodes)
            self.neighborhoods= self.neighborhoods_down
            for i in range(len(self.nodes)):
                self.neighborhoods[i] = np.append(self.neighborhoods_down[i], self.neighborhoods_up[i])
                self.neighborhoods[i] = np.array(self.neighborhoods[i] , dtype=np.int32)
            """
            for i in range(len(self.nodes)):
                nodes = self.nodes[i]
                #neighbs = self.neighborhoods[i]
                if nodes[1]<self.horizon:
                    for j in range(len(self.nodes)) :
                        near_nodes = self.nodes[j]
                        if np.absolute(near_nodes[0]-nodes[0] < self.horizon):
                            if np.absolute((width - near_nodes[1])-nodes[1])< self.horizon:
                                self.neighborhoods[i] = np.append(self.neighborhoods[i], [j])
                                self.neighborhoods[i] = np.array(self.neighborhoods[i] , dtype=np.int32)
            """
	else:
            #Setup empty data on other ranks
            self.neighborhoods = []

        """
        print self.nodes
        #self.create_grid(length, width, 1 )
        if self.rank==0 :
            for i in self.neighborhoods:
                for current_node in self.nodes[i]:
                    if current_node[1] < 0:
                        new_y= width + current_node[1]
                        self.nodes[i] = [current_node[0],new_y]
            print self.nodes
            for i in range(len(self.neighborhoods)):
                for neighbors in self.neighborhoods[i]:
                    xy_of_neighbs = self.nodes[neighbors]
                    if xy_of_neighbs[1]<0.0:
                        self.nodes[neighbors][1] = xy_of_neighbs[1]+width + self.grid_spacing
                    if xy_of_neighbs[1]>width:
                        self.nodes[neighbors][1]=xy_of_neighbs[1]-width - self.grid_spacing
            plt.plot(self.nodes[:,0], self.nodes[:,1], '.')
            for neighb in self.neighborhoods[0]:
                xy_of_neig = self.nodes[neighb]
                plt.plot(xy_of_neig[0], xy_of_neig[1], 'rs')
            for neighb in self.neighborhoods[1300]:
                xy_of_neig = self.nodes[neighb]
                plt.plot(xy_of_neig[0], xy_of_neig[1], 'g^')
            plt.margins(0.1, 0.1)
            plt.show()
        """
        return


    def __init_neighborhood_graph(self):
        """
           Creates the neighborhood ``connectivity'' graph.  This is used to
           load balanced the problem and initialize Jacobian data.
        """

        #Create the standard unbalanced map to instantiate the Epetra.CrsGraph
        #This map has all nodes on the 0 rank processor.
        standard_map = Epetra.Map(self.__global_number_of_nodes,
                len(self.nodes), 0, self.comm)
	#Compute a list of the lengths of each neighborhood list
        num_indices_per_row = np.array([ len(item)
            for item in self.neighborhoods ], dtype=np.int32)
	#Instantiate the graph
        self.neighborhood_graph = Epetra.CrsGraph(Epetra.Copy, standard_map,
                num_indices_per_row, True)
        #Fill the graph
        for rid,row in enumerate(self.neighborhoods):
	    self.neighborhood_graph.InsertGlobalIndices(rid,row)
        #Complete fill of graph
        self.neighborhood_graph.FillComplete()

	return

    def __load_balance(self):
        """Load balancing function."""

        #Load balance
        if self.rank == 0: print "Load balancing neighborhood graph...\n"
        #Create Teuchos parameter list to pass parameters to ZOLTAN for load
        #balancing
        parameter_list = Teuchos.ParameterList()
        parameter_list.set("Partitioning Method","block")
        if not self.verbose:
            parameter_sublist = parameter_list.sublist("ZOLTAN")
            parameter_sublist.set("DEBUG_LEVEL", "0")
        #Create a partitioner to load balance the graph
        partitioner = Isorropia.Epetra.Partitioner(self.neighborhood_graph,
                parameter_list)
        #And a redistributer
        redistributer = Isorropia.Epetra.Redistributor(partitioner)

        #Redistribute graph and store the map
        self.balanced_neighborhood_graph = redistributer.redistribute(
                self.neighborhood_graph)
        self.balanced_map = self.balanced_neighborhood_graph.Map()

	# REMOVED Jason#
	""" Cannot redistribute based on neighborhood graph and
		xy neighborhood graph because there is no
		guarantee the partitioning tools will balance both
		in the same way... """
	#self.xy_balanced_neighborhood_graph = redistributer.redistribute(
        #        self.xy_neighborhood_graph)
        #self.xy_balanced_map = self.balanced_neighborhood_graph.Map()

        #rambod 2D
        self.g_nodes = self.__global_number_of_nodes
        """Assign displacement and pressure indices for each node"""
        Number_of_Global_Variables = 2 * self.g_nodes
        Global_Indices = self.balanced_map.MyGlobalElements()

        XY_Global_Indices = np.zeros(2*len(Global_Indices),dtype = np.int32)

        for index in range(len(Global_Indices)):
            XY_Global_Indices[2*index] = 2*Global_Indices[index]
            XY_Global_Indices[2*index+1]= 2*Global_Indices[index]+1


        XY_list = XY_Global_Indices.tolist()

        #create Epetra Map based on node degrees of Freedom
        self.xy_balanced_map = Epetra.Map(Number_of_Global_Variables,
                XY_list, 0, self.comm)
	#Instantiate the corresponding graph
        self.xy_balanced_neighborhood_graph = Epetra.CrsGraph(Epetra.Copy,
                self.xy_balanced_map,True)
        #fill the XYP vaiable graph
        ### form: [Node N] >>> [X_disp_N, Y_disp_N, Pressure_N] ###
        for index in range(len(Global_Indices)):
            #array of Global indices in neighborhood of each node
            Global_Index = np.asarray(self.balanced_neighborhood_graph
                    .ExtractGlobalRowCopy(Global_Indices[index]))
            #convert global node indices to appropriate xyp indices
            x_index = 2*Global_Index
            x_index = np.array(x_index, dtype=np.int32)
            y_index = 2*Global_Index +1
            y_index = np.array(y_index, dtype=np.int32)

            #Group and sort xyp indices in 1 array
            xy_col_indices = np.sort(np.array([x_index,y_index],
                dtype=np.int32).flatten())
            #insert colums into balanced graph per appropriate rows
            self.xy_balanced_neighborhood_graph.InsertGlobalIndices(
                    2*Global_Indices[index],xy_col_indices)
            self.xy_balanced_neighborhood_graph.InsertGlobalIndices(
                    (2*Global_Indices[index]+1),xy_col_indices)
            #completer fill of balanced grpah per appropriate rows

	self.xy_balanced_neighborhood_graph.FillComplete()
	#create balanced xyp map form balanced xyp neighborhood graph
	self.xy_balanced_map = self.xy_balanced_neighborhood_graph.Map()
        return

    def __init_jacobian(self):
        """
           Initialize Jacobian based on the row and column maps of the balanced
           neighborhood graph.
        """
        xy_graph = self.get_xy_balanced_neighborhood_graph()
	self.__jac = Epetra.CrsMatrix(Epetra.Copy, xy_graph)
        return





    def __init_overlap_import_export(self):
        """
           Initialize Jacobian based on the row and column maps of the balanced
           neighborhood graph.
        """

        balanced_map = self.get_balanced_map()
	ps_balanced_map = self.get_balanced_xy_map()

	overlap_map = self.get_overlap_map()
        ps_overlap_map = self.get_xy_overlap_map()

	self.overlap_importer = Epetra.Import( balanced_map, overlap_map)
	self.overlap_exporter = Epetra.Export(overlap_map,balanced_map)
	self.xy_overlap_importer = Epetra.Import( ps_balanced_map, ps_overlap_map)
        self.xy_overlap_exporter = Epetra.Export(ps_overlap_map , ps_balanced_map)

        return

    def __init_grid_data(self):
        """
           Create data structure needed for doing computations
        """

        #Create some local (to function) convenience variables
        balanced_map = self.get_balanced_map()
	ps_balanced_map = self.get_balanced_xy_map()

	overlap_map = self.get_overlap_map()
        ps_overlap_map = self.get_xy_overlap_map()

	overlap_importer = self.get_overlap_importer()
	ps_overlap_importer = self.get_xy_overlap_importer()

        neighborhood_graph = self.get_balanced_neighborhood_graph()
        xy_neighborhood_graph = self.get_xy_balanced_neighborhood_graph()

        nodes_numb = self.nodes_numb
        horizon = self.horizon


        #Store the unbalanced nodes in temporary x and y position vectors
        if self.rank == 0:
            my_x_temp = self.nodes[:,0]
            my_y_temp = self.nodes[:,1]

	    my_xy_temp = np.vstack( (my_x_temp, my_y_temp) ).T.flatten()
	    my_ps_temp = np.vstack( (0*my_x_temp, 0*my_y_temp) ).T.flatten()
	    #sat = np.linspace(0.4,0.6,len(my_x_temp))
	    #my_ps_temp[1::2] = sat

        else:
            my_x_temp = np.array([],dtype=np.double)
            my_y_temp = np.array([],dtype=np.double)
            my_xy_temp = np.array([],dtype=np.double)
            my_ps_temp = np.array([],dtype=np.double)

        #Create a temporary unbalanced map
        unbalanced_map = Epetra.Map(self.__global_number_of_nodes,
                len(self.nodes), 0, self.comm)

	""" Needed to build the combined unbalanced map to export values
		from head node to all nodes """
        ps_unbalanced_map = Epetra.Map(2*self.__global_number_of_nodes,
                2*len(self.nodes), 0, self.comm)

        #Create the unbalanced Epetra vectors that will only be used to import
        #to the balanced x and y vectors
        my_x_unbalanced = Epetra.Vector(unbalanced_map, my_x_temp)
	my_y_unbalanced = Epetra.Vector(unbalanced_map, my_y_temp)
	my_xy_unbalanced = Epetra.Vector(ps_unbalanced_map, my_xy_temp)
	# ADDED Jason#
	my_ps_unbalanced = Epetra.Vector(ps_unbalanced_map, my_ps_temp)


	#Create the balanced x and y vectors
	my_xy = Epetra.Vector(ps_balanced_map)

	#Create an importer
	ps_importer = Epetra.Import( ps_balanced_map, ps_unbalanced_map )

	#Import the unbalanced data to balanced data

        my_xy.Import(my_xy_unbalanced, ps_importer, Epetra.Insert)
	my_xy_overlap = Epetra.Vector(ps_overlap_map)
        my_xy_overlap.Import(my_xy, ps_overlap_importer, Epetra.Insert)

	#Query the graph to get max indices of any neighborhood graph row on
        #processor (the -1 will make the value correct after the diagonal
        #entries have been removed) from the graph
        my_row_max_entries = neighborhood_graph.MaxNumIndices() - 1

        #Query the number of rows in the neighborhood graph on processor
        my_num_rows = neighborhood_graph.NumMyRows()
	#Allocate the neighborhood array, fill with -1's as placeholders
        my_neighbors_temp = np.ones((my_num_rows, my_row_max_entries),
                dtype=np.int32) * -1
	#Extract the local node ids from the graph (except on the diagonal)
        #and fill neighborhood array
        for rid in range(my_num_rows):
            #Extract the row and remove the diagonal entry
            row = np.setdiff1d(neighborhood_graph.ExtractMyRowCopy(rid),
                    [rid], True)
	    #Compute the length of this row
            row_length = len(row)
            #Fill the neighborhood array
            my_neighbors_temp[rid, :row_length] = row

        #Convert the neighborhood array to a masked array.  This allows for
        #fast computations using numpy. Ragged Python neighborhood lists would
        #prevent this.
        self.my_neighbors = ma.masked_equal(my_neighbors_temp, -1)
        self.my_neighbors.harden_mask()
	#Create distributd vectors needed for the residual calculation
        #(owned only)

	""" pressure and saturation combined and set for import routine """

	my_ps = Epetra.Vector( ps_balanced_map)
	self.residual = my_ps
	self.F_fill = Epetra.Vector( ps_balanced_map)

	ps_importer = Epetra.Import( ps_balanced_map, ps_unbalanced_map )
	my_ps.Import( my_ps_unbalanced, ps_importer, Epetra.Insert )

	my_ps_overlap = Epetra.Vector( ps_overlap_map )
	self.ps_overlap = Epetra.Vector( ps_overlap_map )
        self.residual_overlap = self.ps_overlap
	my_ps_overlap.Import( my_ps, ps_overlap_importer, Epetra.Insert )
	self.F_fill_overlap = Epetra.Vector( ps_overlap_map)

        #List of Global xyp overlap indices on each rank
        ps_global_overlap_indices = ps_overlap_map.MyGlobalElements()
        #Indices of Local x, y, & p overlap indices based on Global indices
        p_local_overlap_indices = np.where(ps_global_overlap_indices%2==0)
        s_local_overlap_indices = np.where(ps_global_overlap_indices%2==1)
        #print p_local_overlap_indices
        #print s_local_overlap_indices
        #ttt.sleep(2)


        #Extract x,y, and p overlap [owned+ghost] vectors
        my_p_overlap = my_ps_overlap[p_local_overlap_indices]
        my_s_overlap = my_ps_overlap[s_local_overlap_indices]

        #List of Global xyp indices on each rnak
        ps_global_indices = ps_balanced_map.MyGlobalElements()
        #Indices of Local x,y,& p indices based on Global indices
        p_local_indices = np.where(ps_global_indices%2==0)
        s_local_indices = np.where(ps_global_indices%2==1)


	my_x = my_xy[p_local_indices]
	my_y = my_xy[s_local_indices]

	my_x_overlap = my_xy_overlap[p_local_overlap_indices]
	my_y_overlap = my_xy_overlap[s_local_overlap_indices]

	#Compute reference position state of all nodes
        self.my_ref_pos_state_x = ma.masked_array(
                my_x_overlap[[self.my_neighbors]] -
                my_x_overlap[:my_num_rows,None],
                mask=self.my_neighbors.mask)
	#
	self.my_ref_pos_state_y = ma.masked_array(
                my_y_overlap[[self.my_neighbors]] -
                my_y_overlap[:my_num_rows,None],
                mask=self.my_neighbors.mask)


        width = self.width
        for i in range(len(self.my_ref_pos_state_y[:,1])):
            for j in range(len(self.my_ref_pos_state_y[1,:])):
                if self.my_ref_pos_state_y[i,j] > (self.horizon):
                    self.my_ref_pos_state_y[i,j] =  self.my_ref_pos_state_y[i,j] - width - self.grid_spacing
                if self.my_ref_pos_state_y[i,j] < -(self.horizon):
                    self.my_ref_pos_state_y[i,j] =  width + self.grid_spacing + self.my_ref_pos_state_y[i,j]
	self.my_ref_pos_state_y = ma.masked_array(self.my_ref_pos_state_y,
                mask=self.my_neighbors.mask)
        ### plotting the neighborhoods to check###
        """
        if self.rank==0:
            plt.plot(self.nodes[:,0], self.nodes[:,1], '.')
            for items in self.my_neighbors[398]:
                if np.equal(np.mod(items,1),0)== True:
                    nearby_points = self.nodes[items]
                    plt.plot(nearby_points[0],nearby_points[1],'o')
            plt.margins(0.1,0.1)
            plt.show()
            """
	#Compute reference magnitude state of all nodes
        self.my_ref_mag_state = (self.my_ref_pos_state_x *
                self.my_ref_pos_state_x + self.my_ref_pos_state_y *
                self.my_ref_pos_state_y) ** 0.5

	#Initialize the volumes
        if self.width==0:
            self.my_volumes = np.ones_like(my_x_overlap,
                dtype=np.double) * self.grid_spacing
	    self.vol = self.grid_spacing
        else:
            self.my_volumes = np.ones_like(my_x_overlap,
                dtype=np.double) * self.grid_spacing * self.grid_spacing
	    self.vol = self.grid_spacing * self.grid_spacing


        #Extract x,y, and p [owned] vectors
        neighbor = self.my_neighbors

        my_p = my_ps[p_local_indices]
        my_s = my_ps[s_local_indices]
        self.saturation_n = my_ps_overlap[s_local_overlap_indices]

	self.my_x = my_x
	self.my_y = my_y
	self.my_x_overlap = my_x_overlap
	self.my_y_overlap = my_y_overlap

	self.my_pressure = my_p
	self.my_saturation = my_s
	self.my_pressure_overlap = my_p_overlap
	self.my_saturation_overlap = my_s_overlap

	self.my_ps = my_ps
	self.my_ps_overlap = my_ps_overlap


	self.my_flow = Epetra.Vector(ps_balanced_map)
        self.my_flow_overlap = Epetra.Vector(ps_overlap_map)




        self.p_local_indices = p_local_indices
        self.s_local_indices = s_local_indices
        self.p_local_overlap_indices = p_local_overlap_indices
        self.s_local_overlap_indices = s_local_overlap_indices

	self.i = 0

	# Establish Boundary Condition #
	balanced_nodes = zip(self.my_x,self.my_y)
	hgs = 0.5 * self.grid_spacing
        gs = self.grid_spacing
	l = self.length
	num_elements = balanced_map.NumMyElements()


        """Right BC with one horizon thickness"""
        x_min_right = np.where(self.my_x >= l-(2.0*gs+hgs))
        x_max_right = np.where(self.my_y <= l+hgs)
        x_min_right = np.array(x_min_right)
        x_max_right = np.array(x_max_right)
	BC_Right_Edge = np.intersect1d(x_min_right,x_max_right)
	BC_Right_Index = np.sort( BC_Right_Edge )
	BC_Right_fill = np.zeros(len(BC_Right_Edge), dtype=np.int32)
	BC_Right_fill_p = np.zeros(len(BC_Right_Edge), dtype=np.int32)
	BC_Right_fill_s = np.zeros(len(BC_Right_Edge), dtype=np.int32)
	for item in range(len( BC_Right_Index ) ):
	    BC_Right_fill[item] = BC_Right_Index[item]
	    BC_Right_fill_p[item] = 2*BC_Right_Index[item]
	    BC_Right_fill_s[item] = 2*BC_Right_Index[item]+1
	self.BC_Right_fill = BC_Right_fill
	self.BC_Right_fill_p = BC_Right_fill_p
	self.BC_Right_fill_s = BC_Right_fill_s

        """ Left BC with one horizon thickness"""
        x_min_left= np.where(self.my_x >= -hgs)[0]
        x_max_left= np.where(self.my_x <= (1.0*gs+hgs))[0]
	BC_Left_Edge = np.intersect1d(x_min_left,x_max_left)
        BC_Left_Index = np.sort( BC_Left_Edge )
	BC_Left_fill = np.zeros(len(BC_Left_Edge), dtype=np.int32)
	BC_Left_fill_p = np.zeros(len(BC_Left_Edge), dtype=np.int32)
	BC_Left_fill_s = np.zeros(len(BC_Left_Edge), dtype=np.int32)
	for item in range(len(BC_Left_Index)):
	    BC_Left_fill[item] = BC_Left_Index[item]
	    BC_Left_fill_p[item] = 2*BC_Left_Index[item]
	    BC_Left_fill_s[item] = 2*BC_Left_Index[item]+1
	self.BC_Left_fill = BC_Left_fill
	self.BC_Left_fill_p = BC_Left_fill_p
	self.BC_Left_fill_s = BC_Left_fill_s

        """ Left BC with two horizon thickness"""
        x_min_left= np.where(self.my_x >= -hgs)[0]
        x_max_left= np.where(self.my_x <= (0.1))[0]
	BC_Left_Edge_double = np.intersect1d(x_min_left,x_max_left)
        BC_Left_Index_double = np.sort( BC_Left_Edge_double )
	BC_Left_fill_double = np.zeros(len(BC_Left_Edge_double), dtype=np.int32)
	BC_Left_fill_p_double = np.zeros(len(BC_Left_Edge_double), dtype=np.int32)
	BC_Left_fill_s_double = np.zeros(len(BC_Left_Edge_double), dtype=np.int32)
	for item in range(len(BC_Left_Index_double)):
	    BC_Left_fill_double[item] = BC_Left_Index_double[item]
	    BC_Left_fill_p_double[item] = 2*BC_Left_Index_double[item]
	    BC_Left_fill_s_double[item] = 2*BC_Left_Index_double[item]+1
	self.BC_Left_fill_double = BC_Left_fill_double
	self.BC_Left_fill_p_double = BC_Left_fill_p_double
	self.BC_Left_fill_s_double = BC_Left_fill_s_double

        """Bottom BC with one horizon thickness"""
        ymin_bottom = np.where(self.my_y >= (-hgs))[0]
        ymax_bottom = np.where(self.my_y <= (2.0*gs+hgs))[0]
        BC_Bottom_Edge = np.intersect1d(ymin_bottom,ymax_bottom)
        BC_Bottom_fill = np.zeros(len(BC_Bottom_Edge), dtype=np.int32)
	BC_Bottom_fill_p = np.zeros(len(BC_Bottom_Edge), dtype=np.int32)
	BC_Bottom_fill_s = np.zeros(len(BC_Bottom_Edge), dtype=np.int32)
        for item in range(len( BC_Bottom_Edge)):
	    BC_Bottom_fill[item] = BC_Bottom_Edge[item]
	    BC_Bottom_fill_p[item] = 2*BC_Bottom_Edge[item]
	    BC_Bottom_fill_s[item] = 2*BC_Bottom_Edge[item]+1
        self.BC_Bottom_fill = BC_Bottom_fill
	self.BC_Bottom_fill_p = BC_Bottom_fill_p
	self.BC_Bottom_fill_s = BC_Bottom_fill_s

        #TOP BC with one horizon thickness
        ymin_top = np.where(self.my_y >= l-(2.0*gs+hgs))[0]
        ymax_top= np.where(self.my_y <= l+hgs)[0]
        BC_Top_Edge = np.intersect1d(ymin_top,ymax_top)
        BC_Top_fill = np.zeros(len(BC_Top_Edge), dtype=np.int32)
	BC_Top_fill_p = np.zeros(len(BC_Top_Edge), dtype=np.int32)
        BC_Top_fill_s = np.zeros(len(BC_Top_Edge), dtype=np.int32)
        for item in range(len( BC_Top_Edge ) ):
	    BC_Top_fill[item] = BC_Top_Edge[item]
	    BC_Top_fill_p[item] = 2*BC_Top_Edge[item]
	    BC_Top_fill_s[item] = 2*BC_Top_Edge[item]+1
        self.BC_Top_fill = BC_Top_fill
	self.BC_Top_fill_p = BC_Top_fill_p
	self.BC_Top_fill_s = BC_Top_fill_s

        global_overlap_indices = (self.get_overlap_map().MyGlobalElements())
        #### setting up the reshaping parameters
        self.sorted_local_indices = np.argsort(global_overlap_indices)
        self.unsorted_local_indices = np.arange(global_overlap_indices.shape[0])[self.sorted_local_indices]

        # x stride
        self.my_x_overlap_stride = (
             np.argmax(my_x_overlap[self.sorted_local_indices])
             )
        self.my_y_overlap_stride = (
             np.argmax(my_y_overlap[self.sorted_local_indices])
             )
        x_max = np.amax(my_x_overlap)
        x_min = np.amin(my_x_overlap)
        y_max = np.amax(my_y_overlap)
        y_min = np.amin(my_y_overlap)
        delta_x = x_max-x_min
        delta_y = y_max-y_min
        #print self.grid_spacing
        x_length = delta_y/self.grid_spacing
        y_length = delta_x/self.grid_spacing
        self.my_y_stride = y_length+1
        return


    def compute_flow(self, ux, flow, uy, flag):
        """
            Computes the peridynamic flow due to non-local pressure
            differentials. Uses the formulation from Kayitar, Foster, & Sharma.
        """
        comm = self.comm
        neighbors = self.my_neighbors
	neighborhood_graph = self.get_balanced_neighborhood_graph()
        volumes = self.my_volumes
        num_owned = neighborhood_graph.NumMyRows()
        horizon = self.horizon
        node_number = neighbors.shape[0]
        neighb_number = neighbors.shape[1]
        # updating previous step's saturation values
        uy_n = self.saturation_n
        ux_n = self.pressure_n
        self.my_velocity_overlap_n[:]

        #### reshape velocities

        # Theses are the sorted and reshaped overlap vectors
	my_ux = ux[self.sorted_local_indices].reshape(int(self.my_y_stride),-1)
	my_uy = uy[self.sorted_local_indices].reshape(int(self.my_y_stride),-1)
	my_ux_n = ux_n[self.sorted_local_indices].reshape(int(self.my_y_stride),-1)
	my_uy_n = uy_n[self.sorted_local_indices].reshape(int(self.my_y_stride),-1)
	"initialing variables with correct size and format"
	term_x = my_ux
	term_y = my_uy

        up_winder = self.up_scaler
        #### Calculation starts here
	current_feshar = my_ux
        print (self.my_ps_overlap[self.BC_Left_fill_p])
        print (self.BC_Right_fill_p)
        ttt.sleep(3)

	current_feshar [1:-1, 1:-1]= self.feshar_const * ((my_ux[1:-1, :-2] - my_ux[1:-1, 2:]) / 2.0 / self.delta_x + (my_uy[:-2,1:-1] - my_uy[2:, 1:-1]) / 2.0 / self.delta_y)

	# Now we'll compute the residual
	self.residual_overlap = (self.my_velocity_overlap_n[:] - self.my_ps_overlap[:])  / self.delta_t
	# u · ∇u term, with central difference approximation of gradient
	term_x[1:-1, 1:-1] = my_ux[1:-1, 1:-1] * (my_ux[1:-1, :-2] - my_ux[1:-1, 2:]) / 2.0 / self.delta_x
	term_y[1:-1, 1:-1] = my_uy[1:-1, 1:-1] * (my_uy[:-2, 1:-1] - my_uy[2:, 1:-1]) / 2.0 / self.delta_y
	# Add these terms into the residual
	self.residual_overlap[:-1:2] = term_x.flatten()[self.unsorted_local_indices]
        self.residual[:-1:2] += self.residual_overlap[:num_owned]
	self.residual_overlap[1::2] = term_y.flatten()[self.unsorted_local_indices]
	self.residual[1::2] += self.residual_overlap[:num_owned]

	#second term, viscous term ,calculation
	term_x[1:-1, 1:-1]= self.visc * ((my_ux[1:-1, 2:] - 2*my_ux[1:-1,1:-1]+my_ux[1:-1,0:-2])/(self.delta_x**2) + ((my_ux[2:,  1:-1]-2*my_ux[1:-1,1:-1]+my_ux[:-2,1:-1])/self.delta_y**2.0))
	term_y[1:-1, 1:-1]= self.visc * ((my_uy[1:-1, 2:] - 2*my_uy[1:-1,1:-1]+my_uy[1:-1,0:-2])/(self.delta_x**2)+ ((my_uy[2:,   1:-1]-2*my_uy[1:-1,1:-1]+my_uy[:-2,1:-1])/self.delta_y**2.0))
	self.residual_overlap[:-1:2] = term_x.flatten()[self.unsorted_local_indices]
	self.residual[:-1:2] += self.residual_overlap[:num_owned]
	self.residual_overlap[1::2] = term_y.flatten()[self.unsorted_local_indices]
	self.residual[1::2] += self.residual_overlap[:num_owned]


	#third term, feshar term, calculation
	term_x[1:-1, 1:-1]= (-1.0/self.rho) * ( current_feshar[1:-1,:-2]-current_feshar[1:-1,2:])/2.0/self.delta_x
	term_y[1:-1, 1:-1]= (-1.0/self.rho) * ( current_feshar[:-2,1:-1]-current_feshar[2:,1:-1])/2.0/self.delta_y
	# Do the rest of the terms
	self.residual_overlap[:-1:2] = term_x.flatten()[self.unsorted_local_indices]
	self.residual[:-1:2] += self.residual_overlap[:num_owned]
	self.residual_overlap[1::2] = term_y.flatten()[self.unsorted_local_indices]
	self.residual[1::2] += self.residual_overlap[:num_owned]


	#fourth term, time derivative calculation
	term_x[1:-1, 1:-1] = (my_ux[1:-1,1:-1] - my_ux_n[1:-1,1:-1]) / self.delta_t
	term_y[1:-1, 1:-1] = (my_uy[1:-1,1:-1] - my_uy_n[1:-1,1:-1]) / self.delta_t
	self.residual_overlap[:-1:2] = term_x.flatten()[self.unsorted_local_indices]
	self.residual[:-1:2] += self.residual_overlap[:num_owned]
	self.residual_overlap[1::2] = term_y.flatten()[self.unsorted_local_indices]
	self.residual[1::2] += self.residual_overlap[:num_owned]



        flow[:] = 0.0
	flow[:2*num_owned] += self.residual
	return

    ###########################################################################
    ####################### NOX Required Functions ############################
    ###########################################################################

    def computeF(self, x, F, flag):
        """
           Implements the residual calculation as required by NOX.
        """
        try:
            overlap_importer = self.get_overlap_importer()
	    ps_overlap_importer = self.get_xy_overlap_importer()

	    neighborhood_graph = self.get_balanced_neighborhood_graph()
	    num_owned = neighborhood_graph.NumMyRows()
	    ux_local_indices = self.p_local_indices
	    uy_local_indices = self.s_local_indices
	    ux_local_overlap_indices = self.p_local_overlap_indices
	    uy_local_overlap_indices = self.s_local_overlap_indices
            #Communicate the pressure (previous or boundary condition imposed)
            #to the worker vectors to be used in updating the flow
	    if self.jac_comp == True:
		self.ps_overlap = x
		x = x[:2*num_owned]

	    if self.jac_comp == False:
		self.ps_overlap.Import(x, ps_overlap_importer,
			Epetra.Insert)

	    my_ux_overlap = self.ps_overlap[ux_local_overlap_indices]
            my_uy_overlap = self.ps_overlap[uy_local_overlap_indices]

	    #Compute the internal flow
            self.compute_flow(my_ux_overlap, self.my_flow_overlap,
                    my_uy_overlap, flag)
	    #Communicate values from worker vectors (owned + ghosts) back to
            #owned only
	    self.my_flow.Export(self.my_flow_overlap, overlap_importer,
		    Epetra.Add)
	    ### PRESSURE BOUNDARY CONDITION & RESIDUAL APPLICATION ###
            self.F_fill_overlap[ux_local_overlap_indices]=self.my_flow_overlap[ux_local_overlap_indices]
            self.F_fill_overlap[uy_local_overlap_indices]=self.my_flow_overlap[uy_local_overlap_indices]
            #Export F fill from [ghost+owned] to [owned]
            # Epetra.Add adds off processor contributions to local nodes

            self.F_fill.Export(self.F_fill_overlap, ps_overlap_importer, Epetra.Add)

            ##update residual F with F_fill
            F[:] = self.F_fill[:]
            #F[self.BC_Left_fill_s_dist] = x[self.BC_Left_fill_s_dist]-1.0
            #F[self.BC_Left_fill_s_double] = x[self.BC_Left_fill_s_double] - 1.0
            F[self.BC_Left_fill_p] = x[self.BC_Left_fill_p] - 0.157
            F[self.BC_Right_fill_p] = x[self.BC_Right_fill_p] - 0.157
            F[self.BC_Right_fill_s] = x[self.BC_Right_fill_s] - 0.0
            F[self.BC_Left_fill_s] = x[self.BC_Left_fill_s] - 0.0
            #F[self.BC_Top_fill_p] = x[self.BC_Top_fill_p] -1000.0
            #F[self.BC_Bottom_fill_p] = x[self.BC_Bottom_fill_p] -0.0
            #F[self.center_fill_s] = x[self.center_fill_s] - 0.0
            #F[self.center_fill_p] = x[self.center_fill_p] - 0.0

            #print my_p_overlap
            self.i = self.i + 1

        except Exception, e:
            print "Exception in PD.computeF method"
            print e

            return False

        return True


    # Compute Jacobian as required by NOX
    def computeJacobian(self, x, Jac):
	try:
	    #print " Jacobian called "
            pass

	except Exception, e:
	    print "Exception in PD.computeJacobian method"
	    print e
	    return False

	return True


    #Getter functions
    def get_balanced_map(self):
        return self.balanced_map

    # ADDED Jason#
    def get_balanced_xy_map(self):
        return self.xy_balanced_map

    def get_overlap_map(self):
        return self.balanced_neighborhood_graph.ColMap()
    def get_xy_overlap_map(self):
        return self.xy_balanced_neighborhood_graph.ColMap()

    def get_overlap_importer(self):
        return self.balanced_neighborhood_graph.Importer()
        #return self.overlap_importer

    def get_xy_overlap_importer(self):
        return self.xy_balanced_neighborhood_graph.Importer()
        #return self.xy_overlap_importer

    def get_overlap_exporter(self):
        return self.balanced_neighborhood_graph.Exporter()
        #return self.overlap_exporter
    def get_xy_overlap_exporter(self):
        return self.balanced_neighborhood_graph.Exporter()
        #return self.ps_overlap_exporter
    def get_balanced_neighborhood_graph(self):
        return self.balanced_neighborhood_graph
    def get_xy_balanced_neighborhood_graph(self):
        return self.xy_balanced_neighborhood_graph
    def get_neighborhood_graph(self):
        return self.neighborhood_graph
    def get_jacobian(self):
        return self.__jac
    def get_solution_pressure(self):
        return self.my_pressure
    #rambod
    def get_solution_saturation(self):
        return self.my_saturation
    def get_x(self):
        return self.my_x
    def get_y(self):
	return self.my_y
    def get_ps_init(self):
	return self.my_ps
    def get_comm(self):
        return self.comm


if __name__ == "__main__":

    def main():
	#Create the PD object
        i=0
        nodes=10
	problem = PD(nodes,10.0)
        comm = problem.comm
	#Define the initial guess
	init_ps_guess = problem.get_ps_init()
   	ps_graph = problem.get_xy_balanced_neighborhood_graph()
        p_local_indices = problem.p_local_indices
        s_local_indices = problem.s_local_indices
        time_stepping = problem.delta_t
        s_local_overlap_indices = problem.s_local_overlap_indices
        p_local_overlap_indices = problem.p_local_overlap_indices
        problem.saturation_n = problem.ps_overlap[s_local_overlap_indices]
        problem.pressure_n = problem.ps_overlap[p_local_overlap_indices]
        ref_pos_state_x = problem.my_ref_pos_state_x
        ref_pos_state_y = problem.my_ref_pos_state_y
        ref_mag_state = problem.my_ref_mag_state
	neighborhood_graph = problem.get_balanced_neighborhood_graph()
        num_owned = neighborhood_graph.NumMyRows()
        neighbors = problem.my_neighbors
        node_number = neighbors.shape[0]
        neighb_number = neighbors.shape[1]
        size_upscaler = (node_number , neighb_number)
        problem.up_scaler = np.ones(size_upscaler)
        horizon = problem.horizon
        volumes = problem.my_volumes
        size = ref_mag_state.shape
        one = np.ones(size)


        """ ################ choose the right kernel function ####### """
        """ for omega = 1/ (r/horizon) """
        omega = one
        omega = one - (ref_mag_state/horizon)
        problem.omega = omega
        linear = 1
        #plt.plot(ref_mag_state[20,:],omega[20,:])
        #plt.show()
        """ omega = 1 """
        omega =one
        problem.omega = omega
        problem.omega = omega
        linear = 0
        """ omega from delgosha """
        #x = ref_mag_state / horizon
        #omega = 34.53* (x**6) +-87.89*(x**5) + 66.976 * (x**4) - 3.9475 * (x**3) - 11.756 * (x**2) + 1.1364 * x + 0.9798
        #problem.omega = omega
        #linear = 2

	ps_overlap_importer = problem.get_xy_overlap_importer()
        ps_overlap_map = problem.get_xy_overlap_map()
        my_ps_overlap = problem.my_ps_overlap
        problem.my_velocity_overlap_n = my_ps_overlap



	#Initialize and change some NOX settings
	nl_params = NOX.Epetra.defaultNonlinearParameters(problem.comm,2)
	nl_params["Line Search"]["Method"] = "Polynomial"
	ls_params = nl_params["Linear Solver"]
	ls_params["Preconditioner Operator"] = "Use Jacobian"
	ls_params["Preconditioner"] = "New Ifpack"
        #Establish parameters for ParaView Visualization
        VIZ_PATH='/Applications/paraview.app/Contents/MacOS/paraview'
        vector_variables = ['displacement']
        scalar_variables = ['pressure','saturation']
        outfile = Ensight('output',vector_variables, scalar_variables,
        problem.comm, viz_path=VIZ_PATH)
        """implement upwinding"""
        if linear ==0 :
            if problem.width==0:
                problem.gamma_c = 2.0 / ((horizon**2.0))
                problem.gamma_p = 1.0 / ((horizon**2.0))

            else:
                problem.gamma_c = 4.0 /(np.pi *(horizon**2.0))
                problem.gamma_p = 2.0 /(np.pi *(horizon**2.0))
        if linear == 1:
            if problem.width==0:
                problem.gamma_c = 6.0 / ((horizon**2.0))
                problem.gamma_p = 3.0 / ((horizon**2.0))

            else:
                problem.gamma_c = 12.0 /(np.pi *(horizon**2.0))
                problem.gamma_p = 6.0 /(np.pi *(horizon**2.0))
        if linear == 2:
            if problem.width==0:
                problem.gamma_c = 15.772870 / ((horizon**2.0))
                problem.gamma_p = 7.886435 / ((horizon**2.0))

            else:
                problem.gamma_c = 31.54574 /(np.pi *(horizon**2.0))
                problem.gamma_p = 15.77287 /(np.pi *(horizon**2.0))
        gamma_c = problem.gamma_c
        gamma_p = problem.gamma_p
        ############ Reading simulations results from previous run ##########
        #if i==0:
        #    for j in range(problem.size):
        #        if problem.rank == j:
        #            pre_sol = np.load('sol_out'+'-'+str(j)+'.npy')
        #            pre_sat = np.load('sat_out'+'-'+str(j)+'.npy')
        #    init_ps_guess[p_local_indices]=pre_sol[p_local_indices]
        #    init_ps_guess[s_local_indices]= pre_sol[s_local_indices]
        #    problem.saturation_n = pre_sat
#
        #x = problem.get_x()
        #print x
        end_range=100
        for i in range(end_range):
            print i
            """ USE Finite Difference Coloring to compute jacobian.  Distinction is made
                    between fdc and solver, as fdc handles export to overlap automatically """
	    #if i>5:
	#	problem.time_stepping =0.5*0.000125
            problem.jac_comp = True
            fdc_pressure = NOX.Epetra.FiniteDifferenceColoring(
                   nl_params, problem, init_ps_guess,
                    ps_graph, False, False)
            fdc_pressure.computeJacobian(init_ps_guess)
            jacobian = fdc_pressure.getUnderlyingMatrix()
            jacobian.FillComplete()
            problem.jac_comp = False
            #Create NOX solver object, solve for pressure and saturation
	    if i<5:
            	solver = NOX.Epetra.defaultSolver(init_ps_guess, problem,
                    problem, jacobian,nlParams = nl_params, maxIters=25,
                    wAbsTol=None, wRelTol=None, updateTol=None, absTol = 5.0e-5, relTol = None)
	    else:
            	solver = NOX.Epetra.defaultSolver(init_ps_guess, problem,
                    problem, jacobian,nlParams = nl_params, maxIters=25,
                    wAbsTol=None, wRelTol=None, updateTol=None, absTol = 5.0e-5, relTol = None)
            solveStatus = solver.solve()
            finalGroup = solver.getSolutionGroup()
            solution = finalGroup.getX()

            #resetting the initial conditions
            init_ps_guess[p_local_indices]=solution[p_local_indices]
            #start from the initial guess of zero
            init_ps_guess[s_local_indices]= solution[s_local_indices]
            #saturation_n = solution[s_local_indices]
            my_ps_overlap.Import( solution, ps_overlap_importer, Epetra.Insert )
            problem.my_velocity_overlap_n = my_ps_overlap
            problem.saturation_n = my_ps_overlap[s_local_overlap_indices]
            pressure_n = my_ps_overlap[p_local_overlap_indices]
            problem.pressure_n = pressure_n
            #plotting the results
            sol_pressure = solution[p_local_indices]
            sol_saturation = solution[s_local_indices]

            #finding velocity at everynode
            ################ Write Date to Ensight Outfile #################
            time = i * problem.delta_t
            outfile.write_geometry_file_time_step(problem.my_x, problem.my_y)

            outfile.write_vector_variable_time_step('displacement',
                                                   [0.0*problem.my_x,0.0*problem.my_y], time)
            outfile.write_scalar_variable_time_step('saturation',
                                                   sol_saturation, time)
            outfile.write_scalar_variable_time_step('pressure',
                                                   sol_pressure, time)
            outfile.append_time_step(time)
            outfile.write_case_file(comm)

            ################################################################
        outfile.finalize()

        #for i in range(problem.size):
        #    if problem.rank == i:
        #        np.save('sol_out'+'-'+str(i),solution)
        #        np.save('sat_out'+'-'+str(i),problem.saturation_n)
        #plotting the results
        #x = problem.get_x()
        #print x
        #s_out = comm.GatherAll(sol_saturation).flatten()
        #p_out = comm.GatherAll(sol_pressure).flatten()
        #x_out = comm.GatherAll(x).flatten()
        #if problem.rank==0:
        #    np.save('s_out'+'-'+str(i),s_out)
        #    np.save('p_out'+'-'+str(i),p_out)
        #    np.save('x_out'+'-'+str(i),x_out)
    main()

