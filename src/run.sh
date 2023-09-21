#!/bin/bash

############################################################################################################
# Step 1: Setup default run parameters

iterations=1000 # Total number of times to run program.
N=32 # Number of simulations being run at the same time.
triggers=10000000
workingDirectory=$(pwd)
outputDirectory=${workingDirectory}/Output



confhelp() {
  echo ""
  echo "Running simulations for NN training"
  echo " "
  echo "Usage: ./run.sh [options]";
  echo " "
  echo " "
  echo "Options:"
  echo "--iterations=[Number oif iterations]"
  echo "    Total number of times to run program. ~1000 is a good number"
  echo " "
  echo "--max-threads=[e.g. 8]"
  echo "    Maximum number of parallel threads."
  echo " "
}


############################################################################################################
# Step 2: extract the command line parameters

# Check for help
for C in "${CMD[@]}"; do
  if [[ ${C} == *-h ]] || [[ ${C} == *-hel* ]]; then
    echo ""
    confhelp
    exit 0
  fi
done

# Overwrite default options with user options:
for C in "${CMD[@]}"; do
  if [[ ${C} == *-i*=* ]]; then
    iterations=`echo ${C} | awk -F"=" '{ print $2 }'`
  elif [[ ${C} == *-m* ]]; then
    N=`echo ${C} | awk -F"=" '{ print $2 }'`
  elif [[ ${C} == *-h ]] || [[ ${C} == *-hel* ]]; then
    echo ""
    confhelp
    exit 0
  fi
done

if [ -d ${outputDirectory} ]; then
  mv ${outputDirectory} ${outputDirectory}.old.$(date  +"%Y%m%d.%H%M%S")
fi
mkdir ${outputDirectory}



############################################################################################################
# Step 3: Run it

# Create copies of source file
counter=1
while  [ $counter -le $iterations ]
do
    values=`python3 angleCalculator.py` # Randomly generated angle values.
    cp ${workingDirectory}/Template.source ${outputDirectory}/Source.${counter}.source # Make a copy of the source file.
    sed -i "s|0  0|${values}|"  ${outputDirectory}/Source.${counter}.source # Edit copy of source file
    sed -i "s|TemplateFileName|${outputDirectory}/TestSource.${counter}|" ${outputDirectory}/Source.${counter}.source
    sed -i "s|TemplateTriggers|${triggers}|" ${outputDirectory}/Source.${counter}.source
    counter=$((counter + 1))
done

sourceFiles=$(find ${outputDirectory} -type f -name "*.source" | sort --version-sort)
echo ${sourceFiles}

count=0
# run cosima on the source files
for filename in ${sourceFiles}; do
    seed=$(od -vAn -N4 -tu4 < /dev/urandom)
    command="cosima -s ${seed} -v 0 -z $filename &> /dev/null &"
    eval $command
    (( ++count ))
    echo "Iteration $count of cosima started"
    if (( count % N == 0 )); then 
      echo "Waiting for a batch of ${N} simulations to finish"
      wait
      echo "$count iterations finished"
    fi
    #sleep 1
done
echo "Waiting for all simulations to finish"
wait

simFiles=$(find ${outputDirectory} -type f -name "*.sim.gz"| sort --version-sort)
echo $simFiles

# Run revan command
count=0
for filename in ${simFiles}; do
    command="revan -g ComptonSphere.geo.setup -c Revan.cfg -f $filename --io -a -n &> /dev/null &"
    eval $command
    echo "Iteration $count of revan started"
    if (( ++count % N == 0 )); then
      echo "Waiting for a batch of ${N} revan runs to finish"
      wait
      echo "$count iterations finished"
    fi
    #sleep 1
done
echo "Waiting for all revan runs to finish"
wait

#rm *sim.gz

# run DataSpaceExtractor.py
traFiles=$(find ${outputDirectory}  -type f -name "*.tra.gz" | sort --version-sort)
echo $traFiles

count=0
for filename in ${traFiles}; do
    pythonCommand="python3 DataSpaceExtractor.py -f $filename -o ComptonSphere.geo.setup &> /dev/null &"
    eval $pythonCommand
    echo "Iteration $count of the extractor started"
    if (( ++count % N == 0 )); then 
      echo "Waiting for a batch of ${N} extractor runs to finish"
      wait
      echo "$count iterations finished"
    fi
    #sleep 1
done
echo "Waiting for all extractor runs to finish"
wait

#rm *tra.gz

#mkdir simulationResults7.23

# Loop to move files to the simulationResults folder
#mv *.pkl simulationResults7.23
#cd simulationResults7.23

# change inc number to begin at 500
#for file in *inc*; do
#    number=$(echo "$file" | sed -n 's/.*inc\([0-9]*\).*/\1/p')
#    new_number=$((number + 500))
#    new_name=$(echo "$file" | sed "s/inc${number}/inc${new_number}/")
#    mv "$file" "$new_name"
#done
#echo "DONE MOVING FILES"
