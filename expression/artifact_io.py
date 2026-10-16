"""Write complete PNG buffers atomically, then decode the actual file for QA."""
import io
import os
from pathlib import Path
from PIL import Image


def write_png_bytes(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        f.write(payload);f.flush();os.fsync(f.fileno())
    with Image.open(temporary) as image:image.load()
    os.replace(temporary,path)
    with Image.open(path) as image:image.load()


def save_png(path,image):
    buffer=io.BytesIO();image.save(buffer,format='PNG')
    write_png_bytes(path,buffer.getvalue())


def save_figure(path,figure,dpi=155):
    buffer=io.BytesIO();figure.savefig(buffer,format='png',dpi=dpi)
    write_png_bytes(path,buffer.getvalue())
