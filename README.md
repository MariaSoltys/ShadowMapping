# ShadowMapping
This project is a Python-based software rasterizer that implements **basic shadow mapping** using custom rendering logic and fragment shaders.

## Project Structure
```
├── main.py              # Main rendering logic: camera/light setup and pipeline control 
├── graphicPipeline.py   # Custom rasterization pipeline and fragment shading (with shadow mapping)
├── camera.py            # Camera class: builds view matrix 
├── projection.py        # Perspective projection matrix 
├── readply.py           # Reads .ply geometry files 
├── suzanne.png          # Texture image 
├── Scene3.ply           # 3D model file 
└── README.md            # This file
```

## How to run
```
python main.py
```
This will open two windows:
    -One shows the shadow map (depth buffer from light's view)
    -The other shows the final render result (with shadows applied)
