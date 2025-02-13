touch README.md
mkdir src
mkdir src/pyclassify
mkdir scripts
mkdir test
mkdir shell
mkdir experiments
touch scr/pyclassify/__init__.py
touch scr/pyclassify/utils.py
touch scripts/run.py
touch shell/submit.sbatch
touch shell/submit.sh
touch experiments/config.yaml
touch requirements.txt
source ~/miniconda3/bin/activate
python -m pip freeze > requirements.txt
wget https://raw.githubusercontent.com/lucaedomosconi/templates/refs/heads/main/devtools_project_submit_file
mv devtools_project_submit_file pyproject.toml
sed -i 's/@NAME/lucam/g' pyproject.toml
sed -i 's/@EMAIL/lmosconi@sissa.it/g' pyproject.toml
sed -i 's/@DESCRIPTION/my_project_devtools/g' pyproject.toml
sed -i 's/@PACKAGEVERSION/pyclassify/g' pyproject.toml
echo "\
# remove data files\
.dat\
.data\
" >> .gitignore

