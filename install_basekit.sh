#!/bin/bash
# based on https://software.intel.com/en-us/articles/oneapi-repo-instructions#apt
# use wget to fetch the Intel repository public key
cd /tmp
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB

# add to your apt sources keyring so that archives signed with this key will be trusted.
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
# remove the public key
rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB

echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"

sudo apt-get update
sudo apt-get install intel-basekit


