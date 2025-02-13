import datetime
import os
import subprocess


def convert_ipynb_to_md(ipynb_file):
    # Change this line to add an underscore
    md_file = "_" + os.path.splitext(os.path.basename(ipynb_file))[0] + ".md"
    md_path = os.path.join(os.path.dirname(ipynb_file), md_file)
    try:
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "markdown",
                ipynb_file,
                "--output",
                md_file,
            ],
            check=True,
        )
        print(f"Converted {ipynb_file} to {md_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {ipynb_file}: {e}")
    except FileNotFoundError:
        print(
            "Error: jupyter nbconvert not found. Please install it using 'pip install nbconvert'."
        )


def get_last_modified_time(file_path):
    return datetime.datetime.fromtimestamp(os.path.getmtime(file_path))


def find_and_convert_ipynb_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb"):
                ipynb_file = os.path.join(root, file)
                # Change this line to add an underscore
                md_file = "_" + os.path.splitext(file)[0] + ".md"
                md_path = os.path.join(root, md_file)

                ipynb_modified = get_last_modified_time(ipynb_file)
                md_modified = (
                    get_last_modified_time(md_path)
                    if os.path.exists(md_path)
                    else datetime.datetime.min
                )

                if ipynb_modified > md_modified:
                    print(f"Converting {ipynb_file} (modified: {ipynb_modified})")
                    convert_ipynb_to_md(ipynb_file)
                else:
                    print(f"Skipping {ipynb_file} (not modified since last conversion)")


def get_valid_directory(use_default=False):
    DEFAULT_DIRECTORY = "./docs/"

    if os.environ.get("MKDOCS_CI") or use_default:
        directory = DEFAULT_DIRECTORY
    else:
        directory = input(
            f"Enter the directory path to search for .ipynb files (default: {DEFAULT_DIRECTORY}): "
        ).strip()

    if directory == "":
        directory = DEFAULT_DIRECTORY

    return os.path.abspath(directory) if os.path.isdir(directory) else DEFAULT_DIRECTORY


if __name__ == "__main__":
    target_directory = get_valid_directory()
    print(f"Searching for .ipynb files in: {target_directory}")
    find_and_convert_ipynb_files(target_directory)
    print("Conversion process completed.")

if __name__ == "<run_path>":
    target_directory = get_valid_directory(use_default=True)
    find_and_convert_ipynb_files(target_directory)
