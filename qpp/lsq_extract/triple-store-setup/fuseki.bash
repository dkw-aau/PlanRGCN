mkdir -p /triplestores
wget -P /triplestores https://dlcdn.apache.org/jena/binaries/apache-jena-fuseki-4.7.0.zip
unzip /triplestores/apache-jena-fuseki-4.7.0
export JENA_HOME=/triplestores/apache-jena-4.6.1
export PATH=$PATH:$JENA_HOME/bin
/triplestores/apache-jena-fuseki/fuseki-server --tdb2 --loc=/pathtotdb2 --port 8080 --timeout=10000000 /Dbpedia