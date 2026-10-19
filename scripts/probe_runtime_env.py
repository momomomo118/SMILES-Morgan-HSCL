import importlib.util
import sys


print(sys.version)
for name in ["torch", "rdkit", "transformers", "sklearn", "pandas", "numpy"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {spec is not None}")
    if spec is not None:
        module = __import__(name)
        version = getattr(module, "__version__", "unknown")
        extra = ""
        if name == "torch":
            extra = f" cuda={module.cuda.is_available()}"
        print(f"{name}_version: {version}{extra}")
