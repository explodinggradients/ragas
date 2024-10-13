import os
import subprocess


def convert_ipynb_to_md(ipynb_file):
    md_file = os.path.splitext(ipynb_file)[0] + ".md"
    try:
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "markdown", ipynb_file], check=True
        )
        print(f"Converted {ipynb_file} to {md_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {ipynb_file}: {e}")
    except FileNotFoundError:
        print(
            "Error: jupyter nbconvert not found. Please install it using 'pip install nbconvert'."
        )


def find_and_convert_ipynb_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb"):
                ipynb_file = os.path.join(root, file)
                convert_ipynb_to_md(ipynb_file)


def get_valid_directory():
    while True:
        DEFAULT_DIRECTORY = "./docs/"
        directory = input(
            f"Enter the directory path to search for .ipynb files (default: {DEFAULT_DIRECTORY}): "
        ).strip()

        if directory == "":
            directory = DEFAULT_DIRECTORY
        if os.path.isdir(directory):
            return os.path.abspath(directory)
        else:
            print("Directory does not exist. Please enter a valid")


if __name__ == "__main__":
    target_directory = get_valid_directory()
    print(f"Searching for .ipynb files in: {target_directory}")
    find_and_convert_ipynb_files(target_directory)
    print("Conversion process completed.")
