#!/bin/bash

# Run install.sh to install Python and required libraries
./install.sh

# Define spinner function
spinner()
{
    local pid=$1
    local delay=0.25
    local spinstr='/-\|'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Wrap the entire script in a subshell and pass its PID to the spinner function
(
    # Perform any setup tasks here
    grep -r "pattern" *

    # Run gather.py to collect data and txt from URLs
    python3 gather.py

    # Wait for 30 seconds before executing the next script
    sleep 30s

    # Run clean.py to clean the data and txt URLs
    python3 clean.py

    # Wait for 30 seconds before executing the next script
    sleep 30s

    # Run train.py to tokenize the data and train a language model
    python3 train.py

) & spinner $!

# Wait for the subshell to finish
wait
