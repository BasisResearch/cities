#!/bin/bash

# This is a convenience script 
# to source the .env_pw file from home
# if you have it

cd ~

if [ -f .env_pw ]; then
    echo ".env_pw found, sourcing..."
    
    source .env_pw
    
    if [ -z "$PASSWORD" ]; then
        echo "PASSWORD variable is not set."
    else
        echo "Sourced .env_pw successfully."
        echo "Password: $PASSWORD"  
    fi
else
    echo ".env_pw not found in the home directory."
fi
