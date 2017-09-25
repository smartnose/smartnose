#!/bin/bash
echo "Installing chrome browser"
echo "adding key"
wget -q -0 - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add - 
echo "set repo"
sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
echo "start installation"
sudo apt-get update 
sudo apt-get install google-chrome-stable

