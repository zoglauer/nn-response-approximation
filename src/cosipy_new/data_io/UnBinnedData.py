# Imports:
import numpy as np
from astropy.table import Table
from astropy.io import fits
import h5py
import time
from cosipy.data_io import DataIO
import gzip
import astropy.coordinates as astro_co
import astropy.units as u
from astropy.coordinates import SkyCoord
from scoords import Attitude
from scoords import SpacecraftFrame
import logging
import sys
import math
from tqdm import tqdm
import subprocess
logger = logging.getLogger(__name__)

class UnBinnedData(DataIO):
 
    def read_tra(self, output_name="unbinned_data", run_test=False):
        
        """
        Reads in MEGAlib .tra (or .tra.gz) file.
        Returns COSI dataset as a dictionary of the form:
        cosi_dataset = {'Energies':erg,
                        'TimeTags':tt,
                        'Xpointings':np.array([lonX,latX]).T,
                        'Ypointings':np.array([lonY,latY]).T,
                        'Zpointings':np.array([lonZ,latZ]).T,
                        'Phi':phi,
                        'Chi local':chi_loc,
                        'Psi local':psi_loc,
                        'Distance':dist,
                        'Chi galactic':chi_gal,
                        'Psi galactic':psi_gal}
        
        Arrays contain unbinned data.
        
        Inputs:
        
        output_name: prefix of output file. 
        
        run_test: This is for unit testing only! Keep False
        unless comparing to MEGAlib calculations. 

        Note: The current code is only able to handle data with Compton events.
              It will need to be modified to handle single-site and pair.    

        Set event_max = None to read all events.
        """
        start_time = time.time()
        
        # Initialise empty lists:
        event_min = 0 
        event_max = 1000
        # Total photon energy
        erg = []
        # Time tag in UNIX time
        tt = []
        # Event Type (CE or PE)
        et = []
        # Compton scattering angle
        phi = []
        # Measured data space angle chi (azimuth direction; 0..360 deg)
        phi_loc = []
        # Measured data space angle psi (polar direction; 0..180 deg)
        theta_loc = []
        # Components of dg (position vector from 1st interaion to 2nd)
        dg_x = []
        dg_y = []
        dg_z = []
        dist = []

        # Define electron rest energy, which is used in calculation
        # of Compton scattering angle.
        c_E0 = 510.9989500015 # keV

        # This is for unit testing purposes only.
        # Use same value as MEGAlib for direct comparison: 
        if run_test == True:
            c_E0 = 510.999

        print("Preparing to read file...")
        
        # Open .tra.gz file:
        if self.data_file.endswith(".gz"):
            f = gzip.open(self.data_file,"rt")
            
            # Need to get number of lines for progress bar.
            # First try fast method for unix-based systems:
            try:
                proc=subprocess.Popen('gunzip -c %s | wc -l' %self.data_file, \
                        shell=True, stdout=subprocess.PIPE)
                num_lines = float(proc.communicate()[0])

            # If fast method fails, use long method, which should work in all cases.
            except:
                print("Initial attempt failed.")
                print("Using long method...")
                g = gzip.open(self.data_file,"rt")
                num_lines = sum(1 for line in g)
                g.close()

        # Open .tra file:
        elif self.data_file.endswith(".tra"):
            f = open(self.data_file,"r")

            try:
                proc=subprocess.Popen('wc -l < %s' %self.data_file, \
                        shell=True, stdout=subprocess.PIPE)
                num_lines = float(proc.communicate()[0])
                
            except:
                print("Initial attempt failed.")
                print("Using long method...")
                g = open(self.data_file,"rt")
                num_lines = sum(1 for line in g)
                g.close()

        else: 
            print()
            print("ERROR: Input data file must have '.tra' or '.gz' extenstion.")
            print()
            sys.exit()
        
        # Read tra file line by line:
        print("Reading file...")
        N_events = 0 # number of events
        pbar = tqdm(total=num_lines) # start progress bar

        for line in f:
         
            this_line = line.strip().split()
            pbar.update(1) # update progress bar

            # Make sure line isn't empty:
            if len(this_line) == 0:
                continue

            # Count the number of events:
            if this_line[0] == "ID":
                N_events += 1

            # Option to only parse a subset of events:
            if event_min != None:
                if N_events < event_min:
                    continue
            if event_max != None:
                if N_events >= event_max:
                    pbar.close()
                    print("Stopping here: only reading a subset of events")
                    break

            # Time tag in Unix time (seconds):
            #if this_line[0] == "TI":
            if this_line[0] == "TI" and len(tt) == len(erg)-1:
                tt.append(float(this_line[1]))
                
            
            # Total photon energy and Compton angle: 
            if this_line[0] == "CE":
                # Compute the total photon energy:
                m_Eg = float(this_line[1]) # Energy of scattered gamma ray in keV
                m_Ee = float(this_line[3]) # Energy of recoil electron in keV
                this_erg = m_Eg + m_Ee
                erg.append(this_erg) 
             
                # Compute the Compton scatter angle due to the standard equation,
                # i.e. neglect the movement of the electron,
                # which would lead to a Doppler-broadening.
                this_value = 1.0 - c_E0 * (1.0/m_Eg - 1.0/(m_Ee + m_Eg))
                this_phi = np.arccos(this_value) # radians
                phi.append(this_phi)
                
            # Event type: 
            if this_line[0] == "ET":
                et.append(this_line[1])
            
            # Interaction position information: 
            if (this_line[0] == "CH"):
                
                # First interaction:
                if this_line[1] == "0":
                    v1 = np.array((float(this_line[2]),\
                            float(this_line[3]),float(this_line[4])))
                
                # Second interaction:
                if this_line[1] == "1":
                    v2 = np.array((float(this_line[2]),
                        float(this_line[3]), float(this_line[4])))
                
                    # Compute position vector between first two interactions:
                    dg = v1 - v2
                    dg_x.append(dg[0])
                    dg_y.append(dg[1])
                    dg_z.append(dg[2])
                
        # Close progress bar:
        pbar.close()
        print("Making COSI data set...")

        # Convert dg vector from 3D cartesian coordinates 
        # to spherical polar coordinates, and then extract distance 
        # b/n first two interactions (in cm), psi (rad), and chi (rad).
        # Note: the resulting angles are latitude/longitude (or elevation/azimuthal).
        conv = astro_co.cartesian_to_spherical(np.array(dg_x), np.array(dg_y), np.array(dg_z))
        dist = conv[0].value 
        psi_loc = conv[1].value 
        chi_loc = conv[2].value

        # Initialize arrays:
        erg = np.array(erg)
        phi = np.array(phi)
        phi_loc = np.array(phi_loc)
        psi_loc = np.array(psi_loc)
        dist = np.array(dist)
        tt = np.array(tt)
        et = np.array(et)
    
        # Rotate psi_loc to colatitude, measured from positive z direction.
        # This is requred for mhealpy input.
        psi_loc = (np.pi/2.0) - psi_loc 
        
        # Define test values for psi and chi local;
        # this is only for comparing to MEGAlib:
        self.psi_loc_test = psi_loc
        self.chi_loc_test = chi_loc

        # Make observation dictionary
        cosi_dataset = {'Energies':erg,
                        'TimeTags':tt,
                        'Phi':phi,
                        'Chi local':chi_loc,
                        'Psi local':psi_loc,
                        'Distance':dist}
        self.cosi_dataset = cosi_dataset

        # Write unbinned data to file (either fits or hdf5):
        print("Saving file...")
        self.write_unbinned_output(output_name) 
        
        return 
 
    def construct_scy(self, scx_l, scx_b, scz_l, scz_b):
    
        """
        Construct y-coordinate of spacecraft/balloon given x and z directions
        Note that here, z is the optical axis
        param: scx_l   longitude of x direction
        param: scx_b   latitude of x direction
        param: scz_l   longitude of z direction
        param: scz_b   latitude of z direction
        """
        
        x = self.polar2cart(scx_l, scx_b)
        z = self.polar2cart(scz_l, scz_b)
    
        return self.cart2polar(np.cross(z,x,axis=0))

    def polar2cart(self, ra, dec):
    
        """
        Coordinate transformation of ra/dec (lon/lat) [phi/theta] polar/spherical coordinates
        into cartesian coordinates
        param: ra   angle in deg
        param: dec  angle in deg
        """
        
        x = np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
        y = np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
        z = np.sin(np.deg2rad(dec))
    
        return np.array([x,y,z])

    def cart2polar(self, vector):
    
        """
        Coordinate transformation of cartesian x/y/z values into spherical (deg)
        param: vector   vector of x/y/z values
        """
        
        ra = np.arctan2(vector[1],vector[0]) 
        dec = np.arcsin(vector[2])
    
        return np.rad2deg(ra), np.rad2deg(dec)

    def write_unbinned_output(self, output_name="unbinned_data"):

        """
        Writes unbinned data file to either fits or hdf5.
        
        output_name: Option to specify name of output file. 
        """

        # Data units:
        units=['keV','s','rad:[glon,glat]','rad:[glon,glat]',
                'rad:[glon,glat]','rad','rad','rad','cm','deg','deg']
            
        # For fits output: 
        if self.unbinned_output == 'fits':
            table = Table(list(self.cosi_dataset.values()),\
                    names=list(self.cosi_dataset.keys()), \
                    meta={'data file':ntpath.basename(self.data_file)})
            table.write("%s.fits" %output_name, overwrite=True)
            os.system('gzip -f %s.fits' %output_name)

        # For hdf5 output:
        if self.unbinned_output == 'hdf5':
            with h5py.File('%s.hdf5' %output_name, 'w') as hf:
                for each in list(self.cosi_dataset.keys()):
                    dset = hf.create_dataset(each, data=self.cosi_dataset[each], compression='gzip')        
    
        return

    def get_dict_from_fits(self,input_fits):

        """Constructs dictionary from input fits file"""

        # Initialize dictionary:
        this_dict = {}
        
        # Fill dictionary from input fits file:
        hdu = fits.open(input_fits,memmap=True)
        cols = hdu[1].columns
        data = hdu[1].data
        for i in range(0,len(cols)):
            this_key = cols[i].name
            this_data = data[this_key]
            this_dict[this_key] = this_data

        return this_dict

    def get_dict_from_hdf5(self,input_hdf5):

        """
        Constructs dictionary from input hdf5 file
        
        input_hdf5: Name of input hdf5 file. 
        """

        # Initialize dictionary:
        this_dict = {}

        # Fill dictionary from input h5fy file:
        hf = h5py.File(input_hdf5,"r")
        keys = list(hf.keys())
        for each in keys:
            this_dict[each] = hf[each][:]

        return this_dict

    def select_data(self, unbinned_data=None, output_name="selected_unbinned_data"):

        """
        Applies cuts to unbinnned data dictionary. 
        Only cuts in time are allowed for now. 
        
        unbinned_data: Unbinned dictionary file. 
        output_name: Prefix of output file. 
        """
        
        print("Making data selections...")

        # Option to read in unbinned data file:
        if unbinned_data:
            if self.unbinned_output == 'fits':
                self.cosi_dataset = self.get_dict_from_fits(unbinned_data)
            if self.unbinned_output == 'hdf5':
                self.cosi_dataset = self.get_dict_from_hdf5(unbinned_data)

        # Get time cut index:
        time_array = self.cosi_dataset["TimeTags"]
        time_cut_index = (time_array >= self.tmin) & (time_array <= self.tmax)
    
        # Apply cuts to dictionary:
        for key in self.cosi_dataset:

            self.cosi_dataset[key] = self.cosi_dataset[key][time_cut_index]

        # Write unbinned data to file (either fits or hdf5):
        self.write_unbinned_output(output_name=output_name)

        return

    def combine_unbinned_data(self, input_files, output_name="combined_unbinned_data"):

        """
        Combines input unbinned data files.
        
        Inputs:
        input_files: List of file names to combine.
        output_name: prefix of output file. 
        """

        self.cosi_dataset = {}
        counter = 0
        for each in input_files:

            print()
            print("adding %s..." %each)
            print()
    
            # Read dict from hdf5 or fits:
            if self.unbinned_output == 'hdf5':
                this_dict = self.get_dict_from_hdf5(each)
            if self.unbinned_output == 'fits':
                this_dict = get_dict_from_fits(each)

            # Combine dictionaries:
            if counter == 0:
                for key in this_dict:
                    self.cosi_dataset[key] = this_dict[key]
            
            if counter > 0:
                for key in this_dict:
                    self.cosi_dataset[key] = np.concatenate((self.cosi_dataset[key],this_dict[key]))
                    
            counter =+ 1
        
        # Write unbinned data to file (either fits or hdf5):
        self.write_unbinned_output(output_name=output_name)

        return

