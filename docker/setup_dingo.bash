#!/bin/bash

# Check if RUN_DIR is provided as a positional argument
if [ $# -ne 1 ]; then
    die "Usage: $0 <RUN_DIR>" 1
fi

source ./setup_variables.bash

log_message "Starting DINGO installation..."

# Check if virtual environment directory exists
if [ -d "$VENV_DIR" ]; then
    log_message "Using existing virtual environment: $VENV_DIR"
    log_message "Python version: $(source "$VENV_DIR/bin/activate" && python --version)"
else
    log_message "Creating new virtual environment: $VENV_DIR"
    if ! python3 -m venv "$VENV_DIR"; then
        die "Failed to create virtual environment at $VENV_DIR" 2
    fi
fi

# Clone repository
rm -rf "$INSTALL_DIR"
if ! mkdir -p "$INSTALL_DIR"; then
    die "Failed to create install directory at $INSTALL_DIR" 3
fi

log_message "Cloning repository from $dingo_repo to $INSTALL_DIR"
if ! git clone "$dingo_repo" "$INSTALL_DIR"; then
    die "Failed to clone repository from $dingo_repo" 4
fi

# Install DINGO
if ! source "$VENV_DIR/bin/activate" || ! cd "$INSTALL_DIR" || ! pip install .; then
    die "Failed to install DINGO" 5
fi

# Setting up the output directory
log_message "Setting up directory structure..."
if ! mkdir -p "$OUTPUT_DIR"; then
    die "Failed to create output directory at $OUTPUT_DIR" 6
fi

log_message "Copying files from $INSTALL_DIR to $OUTPUT_DIR"
if ! cp -r "$INSTALL_DIR"/* "$OUTPUT_DIR/"; then
    die "Failed to copy files to $OUTPUT_DIR" 7
fi

log_message "Creating required subdirectories..."
if ! mkdir -p "$OUTPUT_DIR/training_data" || ! mkdir -p "$OUTPUT_DIR/training"; then
    die "Failed to create subdirectories in $OUTPUT_DIR" 8
fi
