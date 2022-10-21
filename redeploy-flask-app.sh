#!/bin/sh

## change dir to the git repo
cd /home/ubuntu/coralnu

## pull latest code
git fetch --all
git reset --hard origin/master

## update packages
## Python 3.8 is used on the server
python3.8 -m pip3 install -r requirements.txt

## restart service
sudo cp -fr /home/ubuntu/coralnu/ubuntu-conf/gunicorn.service /etc/systemd/system/gunicorn.service
sudo systemctl daemon-reload
sudo service gunicorn restart