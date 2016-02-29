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


from PyTrilinos import Epetra
from PyTrilinos import EpetraExt
from PyTrilinos import Teuchos
from PyTrilinos import Isorropia
from PyTrilinos import NOX

import matplotlib.pyplot as plt
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
        self.comm = Epetra.PyComm()
        self.rank = self.comm.MyPID()
        self.size = self.comm.NumProc()
        self.nodes_numb = num_nodes
	#Print version statement
    
        if self.rank == 0: print("PDD.py version 0.4.0zzz\n")

	# Domain properties
        self.iteration = 0
        self.num_nodes = num_nodes
        self.length = length
        self.time_stepping = 0.001
        self.grid_spacing = float(length) / (num_nodes - 1)
        self.bc_values = bc_values
        self.symm_bcs = symm_bcs
	

        #Default settings and attributes
        if horizon != None:
            self.horizon = horizon
        else:
            self.horizon = 3.15 * self.grid_spacing

        if verbose != None:
            self.verbose = True
        else:
            self.verbose = False

        #Flow properties
        self.counter = 0 
	self.permeability = np.array([[1.0e-3, 0.0],[0.0, 1.0e-3]])
        self.viscosity = 0.1
	self.compressibility = 1.0
        self.density = 1000.0
        self.steps = 3
        self.R = 0.3 #log 2 when 2 is the ration between viscosities

        #Setup problem grid
        self.create_grid(length, width)
        #Find the global family array
        self.get_neighborhoods()
        #Initialize the neighborhood graph
        self.__init_neighborhood_graph()
        #Load balance
        self.__load_balance()
        #Initialize jacobian
        self.__init_jacobian()

	#self.__init_overlap_import_export()

        #Initialize grid data structures
        self.__init_grid_data()
    
    def create_grid(self, length, width):
        """Private member function that creates initial rectangular grid"""

        if self.rank == 0:
            #Create grid, if width == 0, then create a 1d line of nodes
            j = np.complex(0,1)
            if width > 0.0:
                grid = np.mgrid[0:length:self.num_nodes*j,
                        0:width:self.num_nodes*j]
                self.nodes = np.asarray(zip(grid[0].ravel(),grid[1].ravel()), 
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


    def get_neighborhoods(self):
	""" cKDTree implemented for neighbor search """
        
        if self.rank == 0:
            #Create a kdtree to do nearest neighbor search
            tree = scipy.spatial.cKDTree(self.nodes)

            #Get all neighborhoods
            self.neighborhoods = tree.query_ball_point(self.nodes, 
                    r=self.horizon, eps=0.0, p=2)
	else:
            #Setup empty data on other ranks
            self.neighborhoods = []

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
	
	self.ps_overlap_importer = Epetra.Import( ps_balanced_map, ps_overlap_map)


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
	self.F_fill = Epetra.Vector( ps_balanced_map)

	ps_importer = Epetra.Import( ps_balanced_map, ps_unbalanced_map )
	my_ps.Import( my_ps_unbalanced, ps_importer, Epetra.Insert )
	
	my_ps_overlap = Epetra.Vector( ps_overlap_map )
	self.ps_overlap = Epetra.Vector( ps_overlap_map )
	my_ps_overlap.Import( my_ps, ps_overlap_importer, Epetra.Insert )
	self.F_fill_overlap = Epetra.Vector( ps_overlap_map)

        #List of Global xyp overlap indices on each rank
        ps_global_overlap_indices = ps_overlap_map.MyGlobalElements()
        #Indices of Local x, y, & p overlap indices based on Global indices
        p_local_overlap_indices = np.where(ps_global_overlap_indices%2==0)
        s_local_overlap_indices = np.where(ps_global_overlap_indices%2==1)

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
        
	#Compute reference magnitude state of all nodes
        self.my_ref_mag_state = (self.my_ref_pos_state_x * 
                self.my_ref_pos_state_x + self.my_ref_pos_state_y * 
                self.my_ref_pos_state_y) ** 0.5
	
	#Initialize the volumes
        self.my_volumes = np.ones_like(my_x_overlap,
                dtype=np.double) * self.grid_spacing * self.grid_spacing

	self.vol = self.grid_spacing * self.grid_spacing

      
        #Extract x,y, amd p [owned] vectors 
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

        
	self.my_flow = Epetra.Vector(balanced_map)
        self.my_flow_overlap = Epetra.Vector(overlap_map)
	
	
	"Flow equiv. for saturation "
	self.my_trans = Epetra.Vector(balanced_map)
        self.my_trans_overlap = Epetra.Vector(overlap_map)
	

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

        #Above Bottom BC with a horizon thickness
        y_max_abo = np.where(self.my_y <= (5.0 *gs+hgs))
        y_min_abo = np.where(self.my_y >= (2.0*gs+ hgs))
	Abo_Bottom_Edge = np.intersect1d(y_min_abo , y_max_abo)
        Abo_Bottom_Index = np.sort(Abo_Bottom_Edge)
        Abo_Bottom_fill = np.zeros(len(Abo_Bottom_Edge), dtype=np.int32)
	Abo_Bottom_fill_p = np.zeros(len(Abo_Bottom_Edge), dtype=np.int32)
	Abo_Bottom_fill_s = np.zeros(len(Abo_Bottom_Edge), dtype=np.int32)
        for item in range(len( Abo_Bottom_Index ) ):
	    Abo_Bottom_fill[item] = Abo_Bottom_Index[item]
	    Abo_Bottom_fill_p[item] = 2*Abo_Bottom_Index[item]
	    Abo_Bottom_fill_s[item] = 2*Abo_Bottom_Index[item]+1
        self.Abo_Bottom_fill = Abo_Bottom_fill
	self.Abo_Bottom_fill_p = Abo_Bottom_fill_p
	self.Abo_Bottom_fill_s = Abo_Bottom_fill_s
        
        #Below top BC with a horizon thickness
        y_min_bel = np.where(self.my_y >= (l-(5.0 *gs+hgs)))
        y_max_bel = np.where(self.my_y <= (l-(2.0*gs+ hgs)))
	Bel_Top_Edge = np.intersect1d(y_min_bel,y_max_bel)
	Bel_Top_Index = np.sort( Bel_Top_Edge )
        Bel_Top_fill = np.zeros(len(Bel_Top_Edge), dtype=np.int32)
	Bel_Top_fill_p = np.zeros(len(Bel_Top_Edge), dtype=np.int32)
	Bel_Top_fill_s = np.zeros(len(Bel_Top_Edge), dtype=np.int32)
        for item in range(len( Bel_Top_Index ) ):
	    Bel_Top_fill[item] = Bel_Top_Index[item]
	    Bel_Top_fill_p[item] = 2*Bel_Top_Index[item]
	    Bel_Top_fill_s[item] = 2*Bel_Top_Index[item]+1
        self.Bel_Top_fill = Bel_Top_fill
	self.Bel_Top_fill_p = Bel_Top_fill_p
	self.Bel_Top_fill_s = Bel_Top_fill_s

        """Right BC with one horizon thickness"""
        x_min_right = np.where(self.my_x >= l-(2.0*gs+hgs))
        x_max_right = np.where(self.my_y <= l+hgs)
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
        x_max_left= np.where(self.my_x <= (2.0*gs+hgs))[0]
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

        #Bottom BC with one horizon thickness
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

        #center  BC with one horizon radius
        center = 10.0 * (((nodes_numb /2.0)-1.0)/(nodes_numb-1.0))
        min_center = center - horizon
        max_center =center + horizon
        xmin_center=np.where(min_center< self.my_x)[0]
        xmax_center=np.where(max_center > self.my_x)[0]
        ymin_center=np.where(min_center< self.my_y)[0]
        ymax_center=np.where(max_center> self.my_y)[0]
        c1=np.intersect1d(xmin_center,ymin_center)
        c2= np.intersect1d(xmax_center,ymax_center)
        c= np.intersect1d(c1,c2)
        self.center_neighb = c 
        center_neighb_fill_p = np.zeros(len(c), dtype=np.int32)
	center_neighb_fill_s = np.zeros(len(c), dtype=np.int32)

        for item in range(len(c)):
            center_neighb_fill_p[item] = c[item]*2.0
            center_neighb_fill_s[item]=c[item]*2.0+1.0

        self.center_fill_s = center_neighb_fill_s
        self.center_fill_p = center_neighb_fill_p
        self.center_nodes = c
        
        #center  BC with two horizon radius
        center = 10.0 * (((nodes_numb /2.0)-1.0)/(nodes_numb-1.0))
        min_center_2 = center - 2.0* horizon
        max_center_2 =center + 2.0* horizon
        xmin_center_2=np.where(min_center_2< self.my_x)[0]
        xmax_center_2=np.where(max_center_2 > self.my_x)[0]
        ymin_center_2=np.where(min_center_2< self.my_y)[0]
        ymax_center_2=np.where(max_center_2> self.my_y)[0]
        c1_2=np.intersect1d(xmin_center_2,ymin_center_2)
        c2_2= np.intersect1d(xmax_center_2,ymax_center_2)
        c_2= np.intersect1d(c1_2,c2_2)
        self.center_neighb_2 = c_2
        center_2_neighb_fill_p = np.zeros(len(c_2), dtype=np.int32)
	center_2_neighb_fill_s = np.zeros(len(c_2), dtype=np.int32)

        for item in range(len(c_2)):
            center_2_neighb_fill_p[item] = c_2[item]*2.0
            center_2_neighb_fill_s[item]=c_2[item]*2.0+1.0

        self.center_2_fill_s = center_2_neighb_fill_s
        self.center_2_fill_p = center_2_neighb_fill_p
        self.center_nodes_2 =c_2

        return

    def mirror_BC_Top_Bottom(self, x):
        nodes = self.nodes_numb
        ref_index_s = (nodes* 2.0 - 2.0)- 5.0
        ref_index_p = (nodes* 2.0 - 2.0)- 6.0 
        for i in range(len(self.Bel_Top_fill_p)):
            index = self.Bel_Top_fill_p[i]
            if ((index-ref_index_p) % (nodes*2.0) == 0):
                x[index+2]=x[index]
                x[index+4]=x[index-2]
                x[index+6]=x[index-4]
       
        for i in range(len(self.Abo_Bottom_fill_p)):
            index = self.Abo_Bottom_fill_p[i]
            if ((index-6) % (nodes*2.0) == 0):
                x[index-2]=x[index]
                x[index-4]=x[index+2]
                x[index-6]=x[index+4]

        for i in range(len(self.Bel_Top_fill_s)):
            index = self.Bel_Top_fill_s[i]
            if ((index-ref_index_s) % (2.0*nodes) == 0):
                x[index+2]=x[index]
                x[index+4]=x[index-2]
                x[index+6]=x[index-4]
       
        for i in range(len(self.Abo_Bottom_fill_s)):
            index = self.Abo_Bottom_fill_s[i]
            if ((index-7) % (2.0*nodes) == 0):
                x[index-2]=x[index]
                x[index-4]=x[index+2]
                x[index-6]=x[index+4]

        return x 

    def compute_flow(self, pressure, flow, saturation, flag):
        """ 
            Computes the peridynamic flow due to non-local pressure 
            differentials. Uses the formulation from Kayitar, Foster, & Sharma.
        """
        self.counter += 1 
        comm = self.comm 
	#Access the field data
        neighbors = self.my_neighbors
	neighborhood_graph = self.get_balanced_neighborhood_graph()
	ref_pos_state_x = self.my_ref_pos_state_x
        ref_pos_state_y = self.my_ref_pos_state_y
        ref_mag_state = self.my_ref_mag_state
       
        volumes = self.my_volumes
        num_owned = neighborhood_graph.NumMyRows()
        permeability = self.permeability
	compressibility = self.compressibility
        horizon = self.horizon
        density = self.density 
        
        # calling the saturation functions
        saturation_n = self.saturation_n
        #define viscosity dependence on saturation
        # R is the ratio between the two viscosities. We are taking R as 2 here.
        R=self.R 
        size = saturation.shape
        ones = np.ones(size)
        viscos = np.exp(R*(ones-saturation))* (1.0/density)
        pressure_state = ma.masked_array(pressure[neighbors] - 
                pressure[:num_owned,None], mask=neighbors.mask)
        
        #compute the nonlocal permeability from the local constitutive tensor

        ### equation 27 from the NL conversion document ###
        trace = permeability[0,0] + permeability[1,1]
        peri_perm_xx = permeability[0,0]- 1.0 / 4.0 * trace
        peri_perm_xy = permeability[0,1]
        peri_perm_yx = permeability[1,0] 
        peri_perm_yy = permeability[1,1]- 1.0 / 4.0 * trace
        permeability_dot_ref_pos_state_x = (peri_perm_xx * ref_pos_state_x/viscos[:num_owned,None]
                + peri_perm_yx * ref_pos_state_y/viscos[:num_owned,None])

        permeability_dot_ref_pos_state_y = (peri_perm_xy * (ref_pos_state_x/viscos[:num_owned,None])
                + peri_perm_yy * ref_pos_state_y/viscos[:num_owned,None])

        xi_dot_permeability_dot_xi = (permeability_dot_ref_pos_state_x * 
                ref_pos_state_x + permeability_dot_ref_pos_state_y * 
                ref_pos_state_y)

        #Compute the peridynamic flux state
        alpha = 2.0
        scale_factor = 2.0 * (4.0 - alpha) / (np.pi * 
                horizon ** (4.0 - alpha))

        ref_mag_state_invert = (ref_mag_state ** ( 2.0 * alpha )) ** -1.0
        flux_state = (scale_factor * ref_mag_state_invert *
                xi_dot_permeability_dot_xi * pressure_state)
        
        
        flow[:] = 0.0
	flow[:num_owned] += (flux_state * volumes[neighbors]).sum(axis=1)
        

	return 
    
  
    def compute_saturation(self, saturation, trans,pressure , flag):
        #    Computes the peridynamic saturation due to non-local pressure 
        #    differentials. 
	#Access the field data
        neighbors = self.my_neighbors
	neighborhood_graph = self.get_balanced_neighborhood_graph()
	balanced_map = self.get_balanced_map()
        num_owned = neighborhood_graph.NumMyRows()
	ref_pos_state_x = self.my_ref_pos_state_x
        ref_pos_state_y = self.my_ref_pos_state_y
        ref_mag_state = self.my_ref_mag_state
        saturation_n = self.saturation_n
        volumes = self.my_volumes
        permeability = self.permeability
	compressibility = self.compressibility
        horizon = self.horizon
        density = self.density 
        time_stepping = self.time_stepping
        pe= 1000.0
        R=self.R 
        size = saturation.shape
        ones = np.ones(size)
        viscos = np.exp(R*(ones-saturation))* (1.0/density)
        neighb_number = neighbors.shape[1]
        node_number = neighbors.shape[0]
        size_upscaler = (node_number , neighb_number)
        up_scaler = np.ones(size_upscaler)
        #define viscosity dependence on saturation
        # R is the initial ratio between the two viscosities. We are taking R as 2 here.
        #Compute saturation and pressure state
        saturation_state = ma.masked_array(saturation[neighbors]
            -saturation[:num_owned,None], mask=neighbors.mask)
        pressure_state = ma.masked_array(pressure[neighbors] - 
                pressure[:num_owned,None], mask=neighbors.mask)
        saturation_n_state = ma.masked_array(saturation_n[neighbors]
            -saturation_n[:num_owned,None], mask=neighbors.mask)
        viscos_sum = ma.masked_array(viscos[neighbors] + 
                viscos[:num_owned,None], mask=neighbors.mask)

        #Intermediate calculations
        ### equation 26 from the NL conversion document ###
        """
	gamma_denom = ( np.pi * (horizon**2.0) ) ** -1.0
        gamma  =  3.0 * gamma_denom

        term_2_denom = (ref_mag_state ** 2.0) ** -1.0
        term_2_x = gamma *(pressure_state )* (ref_pos_state_x) * term_2_denom
        term_2_y = gamma *(pressure_state )* (ref_pos_state_y) * term_2_denom
        term_2 = term_2_x + term_2_y
        sum_term_2 = ((term_2)*volumes[neighbors]).sum(axis=1)

        term_3_denom = term_2_denom
        term_3_x = gamma * ( saturation_state )* (ref_pos_state_x) * term_3_denom
        term_3_y = gamma * ( saturation_state )* (ref_pos_state_y) * term_3_denom
        term_3 = term_3_x + term_3_y
        sum_term_3 = ((term_3)*volumes[neighbors]).sum(axis=1) 

        denom_23 = (density*viscos[:num_owned])**-1.0
        sum_terms_23 =(sum_term_2 * sum_term_3) * denom_23

	term_4_denom = term_2_denom
        term_4 = (2.0/pe)*(gamma) * saturation_state * term_4_denom
        sum_term_4 = ((term_4)*volumes[neighbors]).sum(axis=1)

        term_contributions =sum_terms_23+sum_term_4 
        """
        scale_2_denom = ( np.pi * (horizon**2.0) ) ** -1.0
        scale_factor_2  =  3.0 * scale_2_denom
        scale_factor_3 = scale_factor_2
        scale_factor_4 = scale_factor_2
        term_2_denom = (ref_mag_state ** 2.0) ** -1.0
        term_2_x = scale_factor_2*(pressure_state )* (ref_pos_state_x) * term_2_denom
        term_2_y = scale_factor_2*(pressure_state )* (ref_pos_state_y) * term_2_denom
        """
        for i in range(num_owned):
            for j in range(neighb_number):
                if(pressure_state[i,j]<=0):
                    up_scaler[i,j] = 0 

        term_2_x = term_2_x * up_scaler
        term_2_y = term_2_y * up_scaler
        """
        sum_term_2_x = ((term_2_x)*volumes[neighbors]).sum(axis=1)
        sum_term_2_y = ((term_2_y)*volumes[neighbors]).sum(axis=1)


        term_3_denom = term_2_denom

        term_3_x = scale_factor_3 * ( saturation_state )* (ref_pos_state_x) * term_3_denom
        sum_term_3_x = ((term_3_x)*volumes[neighbors]).sum(axis=1) 
        
        term_3_y = scale_factor_3 * ( saturation_state ) * (ref_pos_state_y) * term_3_denom
            
        sum_term_3_y = ((term_3_y)*volumes[neighbors]).sum(axis=1)

        sum_terms_23 = (1.0/(density*viscos[:num_owned])) * (sum_term_2_x * sum_term_3_x + sum_term_2_y * sum_term_3_y)

	term_4_denom = term_2_denom
        term_4 = scale_factor_4 * (1/pe) * saturation_state * term_4_denom
        
        sum_term_4 =  ((term_4)*volumes[neighbors]).sum(axis=1)

        term_contributions =  sum_terms_23  + sum_term_4 
        residual=(((saturation[:num_owned] - saturation_n[:num_owned]) / time_stepping )- term_contributions)
        #Integrate nodal flux
        #Sum the flux contribution from j nodes to i node
        trans[:] = 0.0
	trans[:num_owned] += (residual)
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
	    p_local_indices = self.p_local_indices 
	    s_local_indices = self.s_local_indices 
	    p_local_overlap_indices = self.p_local_overlap_indices 
	    s_local_overlap_indices = self.s_local_overlap_indices 
            #Communicate the pressure (previous or boundary condition imposed)
            #to the worker vectors to be used in updating the flow
	    if self.jac_comp == True:
		self.ps_overlap = x
		x = x[:2.0*num_owned]

	    if self.jac_comp == False:
		self.ps_overlap.Import(x, ps_overlap_importer, 
			Epetra.Insert)
            
	    my_p_overlap = self.ps_overlap[p_local_overlap_indices]
            my_s_overlap = self.ps_overlap[s_local_overlap_indices]
            my_p = self.my_ps[self.p_local_indices]
            
	    #Compute the internal flow
            self.compute_flow(my_p_overlap, self.my_flow_overlap, 
                    my_s_overlap, flag)
            
            #compute saturation field
            self.compute_saturation(my_s_overlap, self.my_trans_overlap, my_p_overlap ,flag)
	    #Communicate values from worker vectors (owned + ghosts) back to 
            #owned only
	    self.my_flow.Export(self.my_flow_overlap, overlap_importer, 
		    Epetra.Add) 
            self.my_trans.Export(self.my_trans_overlap,overlap_importer,
                   Epetra.Add)
	    ### PRESSURE BOUNDARY CONDITION & RESIDUAL APPLICATION ### 
            self.F_fill_overlap[p_local_overlap_indices]=self.my_flow_overlap
            self.F_fill_overlap[s_local_overlap_indices]=self.my_trans_overlap
            #Export F fill from [ghost+owned] to [owned]
            # Epetra.Add adds off processor contributions to local nodes
            

            self.F_fill.Export(self.F_fill_overlap, ps_overlap_importer, Epetra.Add)

            #update residual F with F_fill
            F[:] = self.F_fill[:]

            #F[self.BC_Left_fill_s] = x[self.BC_Left_fill_s] - 0.0
            F[self.BC_Left_fill_s] = x[self.BC_Left_fill_s] - 1.0
            #F[self.BC_Right_fill_s] = x[self.BC_Right_fill_s] - 0.0
            #F[self.BC_Top_fill_s] = x[self.BC_Top_fill_s] -0.0
            F[self.BC_Left_fill_p] = x[self.BC_Left_fill_p] - 1000.0
            F[self.BC_Right_fill_p] = x[self.BC_Right_fill_p] - 0.0
            #F[self.BC_Top_fill_p] = x[self.BC_Top_fill_p] -0.0
            #F[self.BC_Bottom_fill_p] = x[self.BC_Bottom_fill_p] - 0.0
            #F[self.BC_Left_fill_s] = x[self.BC_Left_fill_s] - 1.0
            #F[self.BC_Right_fill_s] = x[self.BC_Right_fill_s] - 0.0
            #F[self.Abo_Bottom_fill_s] = x[self.Abo_Bottom_fill_s] - 1.0 
            #F[self.Bel_Top_fill_s] = x[self.Bel_Top_fill_s] -1.0 
            #F[self.center_fill_s] = x[self.center_fill_s] - 1.0 
            #F[self.center_fill_p] = x[self.center_fill_p] - 1000.0 

            x = self.mirror_BC_Top_Bottom(x)

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
        #return self.ps_overlap_importer

    def get_overlap_exporter(self):
        return self.balanced_neighborhood_graph.Exporter()
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
        nodes=200
	problem = PD(nodes,10)
        comm = problem.comm 
        num_owned = problem.neighborhood_graph.NumMyRows()

	#Define the initial guess
	init_ps_guess = problem.get_ps_init()
   	ps_graph = problem.get_xy_balanced_neighborhood_graph()
        p_local_indices = problem.p_local_indices
        s_local_indices = problem.s_local_indices 
        time_stepping = problem.time_stepping
        init_s = init_ps_guess[ s_local_indices ] 
        s_local_overlap_indices = problem.s_local_overlap_indices 
        problem.saturation_n = problem.ps_overlap[s_local_overlap_indices]
        saturation_n = problem.saturation_n
        init_s = init_ps_guess[s_local_indices]
	
	ps_overlap_importer = problem.get_xy_overlap_importer()
        ps_overlap_map = problem.get_xy_overlap_map()
        my_ps_overlap = problem.my_ps_overlap

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
        problem.iteration=0
        end_range = 10005
        for problem.iteration in range(end_range):
            i = problem.iteration
            print i
            graph = problem.get_balanced_neighborhood_graph()
            balanced_map = problem.get_balanced_map()

            """ USE Finite Difference Coloring to compute jacobian.  Distinction is made 
                    between fdc and solver, as fdc handles export to overlap automatically """

            problem.jac_comp = True
            fdc_pressure = NOX.Epetra.FiniteDifferenceColoring(
                   nl_params, problem, init_ps_guess, 
                    ps_graph, False, False)
            fdc_pressure.computeJacobian(init_ps_guess)
            jacobian = fdc_pressure.getUnderlyingMatrix()
            jacobian.FillComplete()
            problem.jac_comp = False
            #Create NOX solver object, solve for pressure and saturation  
            solver = NOX.Epetra.defaultSolver(init_ps_guess, problem, 
                    problem, jacobian,nlParams = nl_params, maxIters=10,
                    wAbsTol=None, wRelTol=None, updateTol=None, absTol = 5.0e-5, relTol = 2.0e-9)
            solveStatus = solver.solve()
            finalGroup = solver.getSolutionGroup()
            solution = finalGroup.getX()

            #resetting the initial conditions
            init_ps_guess[p_local_indices]=solution[p_local_indices]
            #start from the initial guess of zero 
            init_ps_guess[s_local_indices]= init_s
            saturation_n = solution[s_local_indices]
            my_ps_overlap.Import( solution, ps_overlap_importer, Epetra.Insert )
            problem.saturation_n = my_ps_overlap[s_local_overlap_indices]
            
            #plotting the results 
 
            sol_pressure = solution[p_local_indices]
            sol_saturation = solution[s_local_indices]
            x = problem.get_x() 
            y = problem.get_y() 
            x_plot = problem.comm.GatherAll( x )
            y_plot = problem.comm.GatherAll( y )


            sol_p_plot = problem.comm.GatherAll( sol_pressure )
            sol_s_plot = problem.comm.GatherAll( sol_saturation )
            x_plot = comm.GatherAll(x).flatten()
            y_plot = comm.GatherAll(y).flatten()

            if problem.rank==0 : 
                if (i==30 or i==50 or i==100 or i==10000):
                    plt.scatter( x_plot,y_plot, marker = 's', linewidth='0', c = sol_p_plot, s = 50)
                    plt.colorbar()
                    plt.title('Pressure')
                    plt.show()
                    #plt.scatter( x,y, marker = 's', c = sol_saturation, s = 50 )
                    plt.scatter( x_plot,y_plot, marker = 's', linewidth='0', c = sol_s_plot, s = 50 )
                    plt.colorbar()
                    plt.title('Saturation')
                    plt.show()
            if (i==30):
                sol_pressure = solution[p_local_indices]
                sol_saturation = solution[s_local_indices]
                timer = 2.0 
                ################ Write Date to Ensight Outfile #################
                outfile.write_geometry_file_time_step(problem.my_x, problem.my_y)
                outfile.write_scalar_variable_time_step('saturation', 
                                                       sol_saturation, timer)
                outfile.write_scalar_variable_time_step('pressure', 
                                                       sol_pressure, timer)
                outfile.write_vector_variable_time_step('displacement', 
                                                       [0.0*problem.my_x,0.0*problem.my_y], timer)
                outfile.append_time_step(timer)
                outfile.write_case_file(comm)

                ################################################################

                outfile.finalize()
    main()
