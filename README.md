#### Active Vision Experiments
  
To install you need the uv python package / virtual env manager and the pyproject.toml and uv.lock files should do the rest for dependencies.
And  you need ffmpeg on your development environment PATH.

See all options with `python src/auga/cli.py run --help`

To list all the available Atari environments:
``` bash
    python src/auga/cli.py list-envs
```
or in ipython:    
``` python
    import gymnasium as gym
    list(gym.envs.registry.keys())
```

