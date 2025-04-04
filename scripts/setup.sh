#!bin/bash
apt-get update && apt-get upgrade -y

apt-get install python3 -y
apt-get install python3-pip -y
apt-get install tmux -y

#apt install libcairo2-dev pkg-config python3-dev -y

pip3 install torch==2.1.2 --index-url https://download.pytorch.org/whl/test/cpu
pip3 install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.2+cpu.html
pip3 install torch-sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.2+cpu.html
pip3 install torch_geometric==2.4.0
pip3 install torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/test/cpu
pip3 install torchsummary==1.5.1
pip3 install torchvision==0.16.2 --index-url https://download.pytorch.org/whl/test/cpu
pip3 install -r /PlanRGCN/requirements.txt

#java installation
apt-get -y install maven
apt-get -y install openjdk-17-jdk openjdk-17-jre
#mvn package -f "/PlanRGCN/PlanRGCN/qpe/pom.xml"
#mvn install:install-file -Dfile=/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar -DpomFile=/PlanRGCN/PlanRGCN/qpe/pom.xml

apt-get install graphviz graphviz-dev -y

#pip3 install pyclustering # for baseline