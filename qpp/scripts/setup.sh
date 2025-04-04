pip install pandas seaborn scikit-learn jupyterlab
#pip install -e data_analysis/analysis/
pip install -e /qpp/data_analysis/analysis/
apt-get -y install maven
apt install openjdk-17-jdk openjdk-17-jre -y
(cd jars && mvn install:install-file -Dfile=GMT.jar -DgroupId=com.example -DartifactId=gmt -Dversion=1.0 -Dpackaging=jar)
mvn install:install-file -Dfile=jars/clust4j-1.2.4-SNAPSHOT.jar -DgroupId=com.clust4j -DartifactId=clust4j -Dversion=1.2.4 -Dpackaging=jar
pip install pyclustering
#pip install tensorflow-cpu
#pip install tensorflow
#pip install -e qpp_new/
pip install --force-reinstall tensorflow==2.14.1
pip install -e qpp_new/
