# Extract the data space parameters from the event data

import ROOT as M
import argparse
import os
import pickle
import re

# Load MEGAlib into ROOT
M.gSystem.Load("$(MEGALIB)/lib/libMEGAlib.so")

# Initialize MEGAlib
G = M.MGlobal()
G.Initialize()

# Parse the arguments
parser = argparse.ArgumentParser(description="Extract the data space parameters from the event data")
parser.add_argument('-f', '--file', default='Test.id1.tra.gz', help='The tra file')
parser.add_argument('-o', '--geometry', default='${MEGALIB}/simulations/ComptonSphere.geo.setup')
parser.add_argument('-c', '--integers', type=int, help='an integer')

args = parser.parse_args()

N = args.integers

GeometryName = args.geometry
GeometryName = os.path.expandvars(GeometryName)
if not os.path.exists(GeometryName):
  print("Unable to read geometry " + GeometryName + " - Aborting!")
  quit()

FileName = args.file
FileName = os.path.expandvars(FileName)
if not os.path.exists(FileName):
  print("Unable to read data file " + FileName + " - Aborting!")
  quit()

# Load geometry:
Geometry = M.MDGeometryQuest()
if Geometry.ScanSetupFile(M.MString(GeometryName)) == True:
  print("Geometry " + GeometryName + " loaded!")
else:
  print("Unable to load geometry " + GeometryName + " - Aborting!")
  quit()

# Open the tra reader
Reader = M.MFileEventsTra()
if Reader.Open(M.MString(FileName)) == False:
  print("Unable to open file " + FileName + ". Aborting!")
  quit()

# and populate the event selector with it
Selector = M.MEventSelector()
# add some event selections here

combinedData = [[] for i in range(6)]

# Parse the tra file and all events passing the event selector are stored in the new file
while True:
  Event = Reader.GetNextEvent()
  if not Event:
    break

  if Selector.IsQualifiedEventFast(Event) == True:
    if Event.GetEventType() == M.MPhysicalEvent.c_Compton:
      # Extract the 4 parameters of the data space
      ScatterAngle = Event.Phi() * M.c_Deg
      dir = -Event.Dg()
      Theta = dir.Theta() * M.c_Deg
      Phi = dir.Phi() * M.c_Deg
      Distance = Event.FirstLeverArm()
      Energy = Event.Ei()
      combinedData[0].append(Event.GetId())
      combinedData[1].append(Energy)
      combinedData[2].append(Theta)
      combinedData[3].append(Phi)
      combinedData[4].append(ScatterAngle)
      combinedData[5].append(Distance)

      # do what you need to do with it, e.g. fill your own data space
      # print("Event {}: scatterangle={} deg, theta={} deg, phi={} deg, distance={} cn".format(Event.GetId(), ScatterAngle, Theta, Phi, Distance))

  M.SetOwnership(Event, True)

pickleFileName = FileName + ".pkl"

open_file = open(pickleFileName, "wb")
pickle.dump(combinedData, open_file)
open_file.close()

# Test functioning
#open_file = open(pickleFileName, "rb")
#loaded_list = pickle.load(open_file)
#open_file.close()

print("Done")
