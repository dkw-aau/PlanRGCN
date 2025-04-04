if [ $1 == "init" ]
then
    jupyter notebook --generate-config -y
    printf "\nc.NotebookApp.allow_origin = '*'\nc.NotebookApp.ip = '0.0.0.0'\n" >> /root/.jupyter/jupyter_notebook_config.py
elif [ $1 == "install" ]
then
    pip3 install notebook
    jupyter notebook --generate-config -y
    printf "\nc.NotebookApp.allow_origin = '*'\nc.NotebookApp.ip = '0.0.0.0'\n" >> /root/.jupyter/jupyter_notebook_config.py
fi
mkdir -p ../notebooks
jupyter-lab --ip 0.0.0.0 --no-browser --port=80 --allow-root notebooks/