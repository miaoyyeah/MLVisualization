@echo off

echo Installing backend dependencies...
call venv\Scripts\activate
pip install -r ".\requirements.txt"

cd ".\MLVisualizer"

echo Starting Django backend server...


start cmd /k "python manage.py migrate"
start cmd /k "python manage.py runserver"

echo Starting frontend server...
cd "..\app-name"

start cmd /k "npm install"
start cmd /k "npm run dev"
