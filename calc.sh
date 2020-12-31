sudo apt update -y
sudo apt install python3-pip -y
sudo apt install awscli
cd OpenModelProjectBackend
pip3 install pandas surveyweights tqdm joblib
mkdir pkls
python3 precalculate.py
aws configure