from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import struct

def read_idx(filename):
    with open(filename, 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]

        if magic == 0x00000803:  # images
            num_images, rows, cols = struct.unpack('>III', f.read(12))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num_images, rows, cols)

        elif magic == 0x00000801:  # labels
            num_labels = struct.unpack('>I', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(num_labels)

        else:
            raise ValueError(f'Unknown magic number: {magic}')
            

def preprocess_image(filepath):
    # Open image, convert to grayscale
    img = Image.open(filepath).convert('L')  # 'L' mode = grayscale

    # Invert if background is black
    img = ImageOps.invert(img)

    # Resize to 28x28
    img = img.resize((28, 28))

    # Convert to NumPy array and normalize
    img_array = np.array(img) / 255.0  # shape: (28, 28), values in [0,1]

    # Flatten to vector if needed
    img_vector = img_array.reshape(-1, 1)  # shape: (784, 1)

    return img_array, img_vector
    
"""    
import regex as re
import pydoc
def betterhelp():
    for i, match in enumerate(re.finditer('imshow', pydoc.render_doc('matplotlib.pyplot'), re.IGNORECASE)):
        print(pydoc.render_doc('matplotlib.pyplot')[max(0, match.start()):match.end()+200])
        
import inspect

def smart_help(obj, show_private=False, top_n=10):
    members = inspect.getmembers(obj)
    filtered = [m for m in members if callable(m[1]) and (show_private or not m[0].startswith("_"))]
    print(f"Top {min(top_n, len(filtered))} functions in {obj.__name__ if hasattr(obj, '__name__') else obj}:")
    for name, func in filtered[:top_n]:
        doc = inspect.getdoc(func)
        doc_summary = doc.splitlines()[0] if doc else "No docstring."
        print(f" - {name}(): {doc_summary}")
"""