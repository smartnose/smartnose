#!/bin/bash
# Install Docker
curl -fsSL https://get.docker.com/ | sh
# Add your user name to the permission group
echo 'Input your user name to add to docker permission group'
read user_name
sudo usermod -aG docker $user_name
# Log out and log back in
# Verify docker is installed correctly
docker --version
 
# Install docker compose as root user
sudo -i
curl -L https://github.com/docker/compose/releases/download/1.8.0/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose
# Apply executable permissions to the binary
chmod +x /usr/local/bin/docker-compose
# Log out
exit
# Verify the installation
docker-compose --version
