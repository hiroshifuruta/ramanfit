import os
import sys
from PyInstaller.__main__ import run as pyinstaller_run

def create_spec_file(app_name, additional_data=[]):
    """
    Create a spec file for the Streamlit app.
    
    Parameters:
    - app_name: The main python file of the Streamlit app (e.g., 'app.py').
    - additional_data: A list of tuples. Each tuple should have two elements:
      ('path/to/source', 'path/to/destination')
    """
    app_name_without_ext = os.path.splitext(app_name)[0]
    data_str = repr(additional_data + [('streamlit/static', 'streamlit/static')])  # Include Streamlit static files

    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(['{app_name}'],
             pathex=[],
             binaries=[],
             datas={data_str},
             hiddenimports=[
                'streamlit', 
                'pandas', 
                'numpy', 
                'pydeck', 
                'altair', 
                'importlib_metadata', 
                'importlib.metadata'
             ],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='{app_name_without_ext}',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,  # Set to True to see console output for debugging
          icon=None,
          runtime_tmpdir=None,
          exclude_binaries=False,
          win_private_assemblies=False)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='{app_name_without_ext}')
    """
    spec_file = f"{app_name_without_ext}.spec"
    with open(spec_file, 'w') as spec:
        spec.write(spec_content)
    return spec_file

def convert_streamlit_to_exe(app_name, additional_data=[]):
    """
    Convert a Streamlit app to an executable using PyInstaller.
    
    Parameters:
    - app_name: The main python file of the Streamlit app (e.g., 'app.py').
    - additional_data: A list of tuples where each tuple is ('source', 'destination').
      This is used to include additional data files and directories in the executable.
    """
    # Create the spec file
    spec_file = create_spec_file(app_name, additional_data)

    # Run PyInstaller with the spec file
    pyinstaller_run([spec_file, '--noconfirm', '--clean'])

if __name__ == "__main__":
    # Example usage: python convert_to_exe.py app.py
    if len(sys.argv) < 2:
        print("Usage: python convert_to_exe.py <Streamlit_App.py> [Data_Dir1:Target_Dir1 Data_Dir2:Target_Dir2 ...]")
        sys.exit(1)
    
    app_name = sys.argv[1]
    additional_data = []

    # Parse additional data directories if provided
    for data_arg in sys.argv[2:]:
        if ':' in data_arg:
            source, destination = data_arg.split(':', 1)
            additional_data.append((source, destination))
        else:
            print(f"Invalid format for additional data: {data_arg}. Use source:destination format.")
            sys.exit(1)
    
    # Convert the Streamlit app to an executable
    convert_streamlit_to_exe(app_name, additional_data)