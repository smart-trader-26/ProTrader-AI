set -e
PY="c:/Users/divya/Desktop/finance/.venv/Scripts/python.exe"
echo "########## ic_analysis ##########"
"$PY" ic_analysis.py
echo "########## mega_run ##########"
"$PY" mega_run.py
echo "########## final_run ##########"
"$PY" final_run.py
echo "########## DONE ##########"
