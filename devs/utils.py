import os
import ast
import sys
import subprocess
import importlib.util


class RequirementsInstaller:
    """
    A class that installs required packages based on the imports in a given Python file.

    Args:
        filepath (str): The path to the Python file.

    Methods:
        get_imports(): Returns a list of all the imports in the Python file.
        install_packages(): Installs the required packages using pip.
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def get_imports(self):
        with open(self.filepath, 'r') as file:
            root = ast.parse(file.read())

        # Collect all module names from 'import' statements
        imports = [node.names[0].name.split('.')[0] for node in ast.walk(root) if isinstance(node, ast.Import)]
        # Collect all module names from 'from ... import' statements
        import_froms = [node.module.split('.')[0] for node in ast.walk(root) if isinstance(node, ast.ImportFrom) and node.module is not None]
        all_imports = set(imports + import_froms)

        # Filter out modules that are either in the standard library or already installed
        third_party_imports = [name for name in all_imports if not importlib.util.find_spec(name)]
        return third_party_imports

    def install_packages(self):
        packages = self.get_imports()

        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f'Successfully installed {package}')
            except subprocess.CalledProcessError:
                print(f'Failed to install {package}. Please check if the package name is correct.')


def install_requirements(filepath):
    """
    Installs required packages based on the imports in a given Python file.

    Args:
        filepath (str): The path to the Python file.
    """
    installer = RequirementsInstaller(filepath)
    installer.install_packages()


def create_dir(dir_path):
    """
    Creates a directory if it does not exist.

    Args:
        dir_path (str): The path to the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


if __name__ == "__main__":
    # Use the current file as the filepath
    install_requirements(__file__)
