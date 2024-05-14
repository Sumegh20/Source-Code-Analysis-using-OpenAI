echo [$(date)]: "START"
echo [$(date)]: "Creating conda env with python 3.9"
conda create -p code_env python==3.9 -y
echo [$(date)]: "Activate the conda code_env"
source activate code_env/
echo [$(date)]: "installing dev requirements"
pip install -r requirements.txt
echo [$(date)]: "END"

# Run the file by following command in the terminal
# bash init_setup.sh