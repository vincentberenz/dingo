#!/bin/bash

# Function to display usage message
usage() {
    echo "Usage: $0 <run_directory> <success_flag> <email_config_path>"
    echo "Parameters:"
    echo "  run_directory    Path to the run directory"
    echo "  success_flag     Success status flag"
    echo "  email_config_path Path to email configuration YAML file"
    exit 1
}

# Check if exactly 3 arguments are provided
if [ $# -ne 3 ]; then
    echo "Error: Exactly 3 arguments required"
    usage
fi

# Assign arguments to variables
RUN_DIR="$1"
SUCCESS="$2"
EMAIL_CONFIG_PATH="$3"

EMAIL_CONFIG=$(jq '.' "EMAIL_CONFIG_PATH")


# Extract email configuration
local smtp_server=$(jq -r '.mailhub' <<< "$EMAIL_CONFIG")
local port=$(jq -r '.port' <<< "$EMAIL_CONFIG")
local from_addr=$(jq -r '.root' <<< "$EMAIL_CONFIG")
local auth_user=$(jq -r '.authUser' <<< "$EMAIL_CONFIG")
local auth_pass=$(jq -r '.authPass' <<< "$EMAIL_CONFIG")
local recipients=$(jq -r '.recipients[]' <<< "$EMAIL_CONFIG")


# Set email subject based on success/failure
local subject="dingo toy-npe-model example: $(if [ "$SUCCESS" = "true" ]; then echo "SUCCEEDED"; else echo "FAILED"; fi)"

# Create email body
local body=$(cat <<EOF
DINGO Workflow Report
====================
Start Time: $(if [[ -f "$LOG_FILE" ]]; then head -n 1 "$LOG_FILE" | cut -d']' -f1 | cut -c2-; else echo "N/A"; fi)
End Time: $(date +"%Y-%m-%d %H:%M:%S")
Status: $(if [ "$SUCCESS" = "true" ]; then echo "SUCCESS"; else echo "FAILURE"; fi)
Branch: $(if [[ -d "$INSTALL_DIR" ]]; then cd "$INSTALL_DIR" && git branch --show-current; else echo "N/A"; fi)
Commit: $(if [[ -d "$INSTALL_DIR" ]]; then cd "$INSTALL_DIR" && git rev-parse HEAD; else echo "N/A"; fi)
EOF
      )

# Prepare attachments
local attachments=()
if [[ -f "$LOG_FILE" ]]; then
    attachments+=("-a" "$LOG_FILE")
fi
if [[ -f "$ERROR_FILE" ]]; then
    attachments+=("-a" "$ERROR_FILE")
fi

# URI the mail will be set to
smpt_url="smtps://${auth_user}@${smtp_server}:${port}"

# sending the email
echo "$body" | mutt -e "set smtp_url=${smpt_url}" -e "set smtp_pass=${auth_pass}" -e "set smtp_authenticators='login'" -s "${subject}" ${attachments[@]} -- ${recipients}

