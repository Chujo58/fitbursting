# fitbursting
Definition of fitbursting: doing some black magic and getting fitburst to work on a very complex event that took you half an hour to fit (and then very much celebrate about it and share it to the world).

## What this repo is for?
Running fitburst (hopefully successfully on the first try so you don't spend 2 hours on a single event)

# Installation instructions

Run the install script

```
python install.py
```

## Tips for installing


1. Use ssh instead of http for git clone in install script
    ```
    python install.py --ssh
    ```

2. Use uv to manage python versions: https://docs.astral.sh/uv/getting-started/installation/
        
    This project uses Python 3.12, check if it's installed with `uv python list`
    
    Install Python 3.12: `uv python install 3.12`
    
    Run install script with Python 3.12:
    ```
    uv run --python 3.12 install.py
    ```
