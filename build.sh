#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Navigate to project directory (since we'll move Django project to root)
cd project/ReviveCare

# Collect static files
python manage.py collectstatic --no-input

# Run database migrations
python manage.py migrate
