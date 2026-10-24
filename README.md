# Hemodynamics Platform

Hemodynamics Platform is a Python desktop application for medical-image processing, vascular geometry and mesh preparation, CFD/FSI workflow management, results extraction, and VTK/PyVista visualization. The interface uses PySide2 (Qt 5) and is started from `GUI_V0/HemodynamicsPlatform.py`.

Some modules call external software or executables that are not installed by the Python environment. The main interface can run without them, but the corresponding workflows require those tools to be installed and configured separately.

## Windows installation with Conda (recommended)

The project is a legacy, CPU-only Python 3.7 application. Use the pinned versions below; do not replace Python 3.7 with a newer version unless the source code and dependencies are upgraded together. CUDA and deep-learning frameworks are not required.

### 1. Install Conda

Install a 64-bit Windows version of [Miniconda or Anaconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/windows.html). During installation, the default options are suitable for most users.

After installation, open **Anaconda Prompt** from the Windows Start menu. Use Anaconda Prompt for every command in this guide; a normal Command Prompt may not recognize `conda` until Conda has been initialized.

### 2. Open the project directory

Replace the example path with the folder in which you downloaded or cloned this repository:

```bat
cd /d "C:\Users\YOUR_NAME\Documents\HemodynamicsPlatform"
```

Confirm that you are in the correct directory:

```bat
dir env\environment.yml
dir GUI_V0\HemodynamicsPlatform.py
```

Both commands should display a file. If either says `File Not Found`, correct the project path before continuing.

### 3. Create and activate the environment

```bat
conda env create --file env\environment.yml
conda activate hymo
```

The first command downloads several hundred megabytes and may take a while. If Conda displays `Proceed ([y]/n)?`, type `y` and press Enter. A successful activation changes the beginning of the prompt to `(hymo)`.

For an exact Windows x64 recreation of the environment that passed the import and GUI smoke tests, use the lock file instead of the readable environment file:

```bat
conda env create --file env\environment-win-64.lock.yml
conda activate hymo
```

Check that the correct Python is active:

```bat
where python
python --version
```

`python --version` must report `Python 3.7.16`, and the first path printed by `where python` should contain `hymo`.

### 4. Verify the important imports

Copy this command as one line:

```bat
python -c "import PySide2, vtk, pyvista, pyvistaqt, SimpleITK, cv2, numpy, scipy, skimage, sklearn, pandas, meshio, yaml, radiomics, openpyxl, networkx; print('Environment check passed')"
```

Continue only after it prints `Environment check passed` without an exception.

### 5. Start the interface

```bat
cd GUI_V0
python HemodynamicsPlatform.py
```

Keep Anaconda Prompt open while using the application. On later runs, repeat only these commands from the repository root:

```bat
conda activate hymo
cd GUI_V0
python HemodynamicsPlatform.py
```

## Manual Windows installation if the YAML command fails

The YAML contains the same real packages and order as `env/EnviromentSet.txt`, with Python 3.7-compatible versions taken from the Docker configuration. Conda installs its `pip:` section in one transaction, as the Dockerfile does. Use the commands below only if that transaction fails or if the historical one-command-at-a-time installation is required. They preserve the original sequence while pinning versions so that pip does not select packages that have dropped Python 3.7 support.

First remove an incomplete environment, if one was created:

```bat
conda deactivate
conda env remove --name hymo
```

If Conda reports that the environment does not exist, continue. Create a clean environment and activate it:

```bat
conda create --name hymo python=3.7.16 pip=22.3.1 setuptools=65.6.3 wheel=0.38.4
conda activate hymo
```

If Conda asks whether to proceed, type `y` and press Enter.

Install the packages into the active environment. Run every command and stop if a command reports an error:

```bat
python -m pip install psutil==5.9.4
python -m pip install PySide2==5.15.2.1 shiboken2==5.15.2.1
python -m pip install numpy==1.21.6
python -m pip install SimpleITK==2.0.2
python -m pip install matplotlib==3.4.0
python -m pip install opencv-python==4.5.4.60
python -m pip install scipy==1.7.3
python -m pip install scikit-image==0.19.3
python -m pip install vtk==9.2.6
python -m pip install scikit-learn==1.0.2
python -m pip install pandas==1.3.5
python -m pip install more-itertools==9.1.0
python -m pip install meshio==5.3.4
python -m pip install PyYAML==6.0
python -m pip install pyradiomics==3.0.1
python -m pip install pyvista==0.38.3
python -m pip install pyvistaqt==0.9.1
python -m pip install pyinstaller==5.8.0
python -m pip install openpyxl==3.1.1
python -m pip install networkx==2.6.3
```

Then perform the import check and start the interface as described above.

## Windows troubleshooting

### Qt reports that the `windows` platform plugin cannot be loaded

Do not install PyQt5; this application uses PySide2. With `hymo` activated, first point Qt to the plugin directory bundled with PySide2:

```bat
set "QT_QPA_PLATFORM_PLUGIN_PATH=%CONDA_PREFIX%\Lib\site-packages\PySide2\plugins\platforms"
cd GUI_V0
python HemodynamicsPlatform.py
```

The setting lasts only for the current Anaconda Prompt. If that command fixes the problem and you want the tested file-copy workaround from `env/EnviromentSet.txt`, run:

```bat
xcopy /E /I /Y "%CONDA_PREFIX%\Lib\site-packages\PySide2\plugins\platforms" "%CONDA_PREFIX%\Library\plugins\platforms"
```

`%CONDA_PREFIX%` is filled in by Conda, so these commands work regardless of where Anaconda or Miniconda is installed.

### The window is blank or VTK/OpenGL fails

Force Qt to use software OpenGL for the current prompt, then start the application again:

```bat
set QT_OPENGL=software
python HemodynamicsPlatform.py
```

Also update the graphics driver supplied by the computer or GPU manufacturer.

### `conda` is not recognized

Close the terminal and open **Anaconda Prompt**. If Anaconda Prompt is not present, repair or reinstall Miniconda/Anaconda for the current Windows user.

### A package is missing after installation

Check the active environment before installing anything else:

```bat
conda activate hymo
where python
python --version
```

Do not use `pip install --user`, because that installs packages outside the Conda environment. Recreate the environment using the steps above if it is incomplete.

### The environment already exists

If an earlier installation stopped partway through, recreate it cleanly:

```bat
conda deactivate
conda env remove --name hymo
conda env create --file env\environment.yml
```

## Docker on a Linux desktop (optional)

The Docker image is intended for a Linux host with an X11 desktop. Windows users should use the Conda instructions above. The image uses software OpenGL, so an NVIDIA GPU is not required.

From the repository root, build the image:

```bash
docker build --file env/Dockerfile --tag hemodynamics-platform .
```

Confirm that the Linux host has a display (`echo "$DISPLAY"` should not be empty), temporarily allow the container to connect to X11, and start the GUI:

```bash
xhost +local:docker

docker run --rm --interactive --tty \
  --entrypoint /bin/bash \
  --env DISPLAY="$DISPLAY" \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume "$PWD":/app \
  --workdir /app/GUI_V0 \
  hemodynamics-platform \
  -lc "python HemodynamicsPlatform.py"

xhost -local:docker
```

The `--entrypoint` override is intentional: the image's default entrypoint starts a virtual framebuffer for headless use, whereas the command above connects the application to the host display. The final `xhost` command removes the temporary X11 permission.

To make host data available inside the container, add a mount and select those paths in the application, for example:

```bash
--volume "/path/on/host:/data"
```

## Repository layout

- `GUI_V0/` — main application, modules, and Qt `.ui` files.
- `Functions_AZ/`, `Functions_JZ/`, and `Functions_YC/` — shared processing functions.
- `Base/` — styles and supporting resources.
- `env/environment.yml` — reproducible Conda environment for the legacy application.
- `env/environment-win-64.lock.yml` — exact resolved Windows x64 environment that passed the GUI smoke test.
- `env/EnviromentSet.txt` — original tested Windows installation notes.
- `env/Dockerfile` — optional Linux container environment.
