#!/bin/bash
sudo apt-get update
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-5.6.2.deb
sudo dpkg -i elasticsearch-5.6.2.deb
sudo systemctl enable elasticsearch.service
