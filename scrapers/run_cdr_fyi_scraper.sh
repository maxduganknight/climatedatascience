#!/bin/bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Activate the virtualenv
pyenv activate cdr_fyi_scraper

# Run your script with full path to Python interpreter
"/Users/max/.pyenv/versions/3.12.1/bin/python" /Users/max/Deep_Sky/GitHub/datascience-platform/scrapers/cdr_fyi_scraper.py -s >> /Users/max/Deep_Sky/cron_logfile.log 2>&1

# Deactivate virtualenv
pyenv deactivate