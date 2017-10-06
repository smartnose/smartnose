#!/bin/bash
cert_folder=/usr/local/share/ca-certificates/office365localdev
sudo mkdir ${cert_folder}
sudo cp ./node_modules/browser-sync/lib/server/certs/server.crt ${cert_folder}
sudo chmod 755 ${cert_folder}
sudo chmod 644 ${cert_folder}/server.crt
sudo update-ca-certificates

