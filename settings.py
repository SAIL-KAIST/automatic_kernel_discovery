import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SECRET_KEY = "ABC"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": "/home/anhth/projects/xai/db.sqlite3" # point to the DB of XAI project
    }
}

INSTALLED_APPS = ("web",)

# make sure this point to directory folder
FIGURE_FOLDER = "/home/anhth/projects/xai/static/upload/AutomaticNews/"