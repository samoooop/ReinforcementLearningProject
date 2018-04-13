sudo apt-get install python3-pip git zlib1g-dev cmake swig git mpich libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev
pip3 install virtualenv
cd ~
mkdir .pythonenv
cd .pythonenv
virtualenv .
echo 'source ~/.pythonenv/bin/activate' >> ~/.bashrc
pip3 install gym opencv-python tensorflow-gpu scikit-image
cd ~/Documents
git clone https://github.com/openai/baselines
cd baselines
pip3 install -e .

