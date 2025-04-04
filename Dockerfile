FROM ubuntu:22.04

#COPY PlanRegr/ /PlanRGCN/PlanRegr/
#COPY dist_loader/ /PlanRGCN/dist_loader/
#COPY feat_con_time/ /PlanRGCN/feat_con_time/
#COPY feature_extraction/ /PlanRGCN/feature_extraction/
#COPY feature_representation/ /PlanRGCN/feature_representation/
#COPY graph_construction/ /PlanRGCN/graph_construction/

#COPY notebooks/ /PlanRGCN/notebooks/
#COPY qpe/ /PlanRGCN/qpe/

#COPY sample_checker/ /PlanRGCN/sample_checker/

# Setup ssh for development
RUN apt-get update -y
RUN apt-get install ssh -y
RUN apt-get install rsync -y
RUN apt install openssh-server sudo -y
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 test
RUN  echo 'root:test' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN service ssh start

WORKDIR /PlanRGCN
COPY requirements.txt /PlanRGCN/requirements.txt
COPY scripts/ /PlanRGCN/scripts/
RUN bash scripts/setup.sh


RUN pip3 install -r requirements.txt
RUN pip install JPype1==1.5.0

#No longer needed since local files are used for remote developement
#COPY .git/ /PlanRGCN/.git/
#COPY .gitignore /PlanRGCN/.gitignore
#RUN apt-get install git -y



# Source files
COPY PlanRGCN/ /PlanRGCN/PlanRGCN/
RUN mvn package -f "/PlanRGCN/PlanRGCN/qpe/pom.xml" && mvn install:install-file -Dfile=/PlanRGCN/PlanRGCN/qpe/target/qpe-1.0-SNAPSHOT.jar -DpomFile=/PlanRGCN/PlanRGCN/qpe/pom.xml
COPY Dockerfile /PlanRGCN/Dockerfile
COPY .dockerignore /PlanRGCN/.dockerignore
#COPY .vscode/launch.json /PlanRGCN/.vscode/launch.json
#COPY .vscode/settings.json /PlanRGCN/.vscode/settings.json
#COPY data /PlanRGCN/data
COPY run.sh /PlanRGCN/run.sh
COPY README.md /PlanRGCN/README.md
COPY test_inference_time.py /PlanRGCN/test_inference_time.py
COPY pp_only_qs.py /PlanRGCN/pp_only_qs.py
COPY inductive_query/ /PlanRGCN/inductive_query/
COPY load_balance/ /PlanRGCN/load_balance/
COPY qpp/ /PlanRGCN/qpp/
COPY install_local_modules.sh /PlanRGCN/install_local_modules.sh
RUN bash /PlanRGCN/install_local_modules.sh

COPY utils/ /PlanRGCN/utils/
COPY virt_confs/ /PlanRGCN/virt_feat_conf/

EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]