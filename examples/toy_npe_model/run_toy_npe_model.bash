#!/bin/bash

# This executable runs all the steps of the dingo toy npe model as described here:
# https://dingo-gw.readthedocs.io/en/latest/example_toy_npe_model.html
# Files are created/copied/used in /tmp/dingo_examples/toy_npe_model.
# This executable should be run from the folder dingo/examples/toy_npe_model
# It is assumed dingo has been (pip) installed.

# Checking this script is run from the expected directory,
# exiting with error otherwise
current_dir=$(pwd)
expected_ending="/dingo/examples/toy_npe_model"
if ! [[ "$current_dir" == *"$expected_ending" ]]; then
    echo "this script should be run from ${expected_ending}" >&2
    exit 1
fi

set -e

error_handler() {
  local cmd_name=$1
  local error_message=$2
  echo "Error: ${cmd_name} failed with message: ${error_message}" >&2
  exit 1
}

# Files and folders locations
TOY_FOLDER=/tmp/dingo/toy_npe_example

# starting from scratch
if [ -d "$TOY_FOLDER" ]; then
  # Delete the contents of previous run
  rm -rf "$TOY_FOLDER"/*
fi

# Create a subfolder in /tmp and copying content to it
echo "Copying the content of current folder to ${TOY_FOLDER}"
mkdir -p $TOY_FOLDER || error_handler "mkdir"
cp -r $(pwd)/* $TOY_FOLDER || error_handler "copying files and folder to ${TOY_FOLDER}"
cd ${TOY_FOLDER} || error_handler "moving to ${TOY_FOLDER}"
echo "Working from $(pwd)"
mkdir -p training_data || error_handler "mkdir"
mkdir -p training || error_handler "mkdir"

# Function to print output with command name, tab and grey color
print_output() {
  local cmd_name=$1
  shift
  local output
  output=$("$@" 2>&1) # Capture both stdout and stderr
  local exit_status=$?
  echo "$output" | sed "s/^/[${cmd_name}]\t/" | while IFS= read -r line; do echo -e "\033[90m$line\033[0m"; done
  if [ $exit_status -ne 0 ]; then
    error_handler "$cmd_name" "$output"
  fi
}

# Generate waveform dataset
echo "-- Generating waveform dataset --"
print_output dingo_generate_dataset dingo_generate_dataset --settings waveform_dataset_settings.yaml --out_file training_data/waveform_dataset.hdf5 || error_handler "dingo_generate_dataset"

# Generate ASD dataset
echo "-- Generating ASD dataset --"
print_output dingo_generate_asd_dataset dingo_generate_asd_dataset --settings_file asd_dataset_settings.yaml --data_dir training_data/asd_dataset || error_handler "dingo_generate_asd_dataset"

# Train the network
echo "-- Training --"
print_output dingo_train dingo_train --settings_file train_settings.yaml --train_dir training || error_handler "dingo_train"

# Do inference
echo "-- Performing inference --"
print_output dingo_pipe dingo_pipe GW150914.ini || error_handler "dingo_pipe"
echo "Results can be found in ${TOY_FOLDER}/outdir_GW150914"

# Exit with success
exit 0
