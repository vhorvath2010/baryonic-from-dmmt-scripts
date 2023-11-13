#!/bin/bash

# Set the range of file numbers
start=0
end=103

# Iterate through the range of file numbers
for ((i=start; i<=end; i++)); do
    # Form the file name
    filename="SG256_Full_Graphs_Part_$i.pt"

    # Check if the file exists
    if [ ! -f "$filename" ]; then
        # If the file is missing, print a message
        echo "File $filename is missing."
    fi
done
