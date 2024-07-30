from typing import Iterator, List, Dict, Any, Union, AsyncIterator
from pathlib import Path
import os
import subprocess
import asyncio
import aiofiles
import logging
from langchain_core.documents import Document
from langchain_community.document_loaders.unstructured import UnstructuredBaseLoader
from langchain_community.document_loaders.helpers import detect_file_encodings

# Setup logging
logger = logging.getLogger(__name__)

class RAGASLoader(UnstructuredBaseLoader):
    def __init__(
            self,
            file_path: Union[str, Path, List[Union[str, Path]]],
            mode: str = "single",
            autodetect_encoding: bool = False,
            **unstructured_kwargs: Any
    ):
        super().__init__(mode=mode, **unstructured_kwargs)
        self.autodetect_encoding = autodetect_encoding
        if isinstance(file_path, list):
            self.file_paths = [Path(fp) for fp in file_path]
        else:
            self.file_paths = [Path(file_path)]

    def _get_elements(self, file_path: Path) -> List:
        from unstructured.partition.auto import partition
        try:
            return partition(filename=str(file_path), **self.unstructured_kwargs)
        except Exception as e:
            logger.error("Error in _get_elements for file %s: %s", file_path, e)
            return []

    def _get_metadata(self, file_path: Path) -> Dict:
        try:
            return {
                "source": str(file_path),
                "raw_content": self._read_file(file_path)
            }
        except Exception as e:
            logger.error("Error in _get_metadata for file %s: %s", file_path, e)
            return {"source": str(file_path), "raw_content": ""}

    async def _aget_metadata(self, file_path: Path) -> Dict:
        try:
            return {
                "source": str(file_path),
                "raw_content": await self._aread_file(file_path)
            }
        except Exception as e:
            logger.error("Error in _aget_metadata for file %s: %s", file_path, e)
            return {"source": str(file_path), "raw_content": ""}

    def _read_file(self, file_path: Path) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(file_path)
                for encoding in detected_encodings:
                    try:
                        with file_path.open(encoding=encoding.encoding) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue
                logger.error("Failed to decode %s with detected encodings.", file_path)
                raise RuntimeError(f"Failed to decode {file_path} with detected encodings.")
            else:
                logger.error("Error loading %s due to encoding issue.", file_path, exc_info=e)
                raise RuntimeError(f"Error loading {file_path} due to encoding issue.") from e
        except Exception as e:
            logger.error("Error loading %s due to an unexpected error: %s", file_path, e)
            raise RuntimeError(f"Error loading {file_path} due to an unexpected error: {e}") from e

    async def _aread_file(self, file_path: Path) -> str:
        try:
            async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                return await f.read()
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = await asyncio.to_thread(detect_file_encodings, file_path)
                for encoding in detected_encodings:
                    try:
                        async with aiofiles.open(file_path, mode='r', encoding=encoding.encoding) as f:
                            return await f.read()
                    except UnicodeDecodeError:
                        continue
                logger.error("Failed to decode %s with detected encodings.", file_path)
                raise RuntimeError(f"Failed to decode {file_path} with detected encodings.")
            else:
                logger.error("Error loading %s due to encoding issue.", file_path, exc_info=e)
                raise RuntimeError(f"Error loading {file_path} due to encoding issue.") from e
        except Exception as e:
            logger.error("Error loading %s due to an unexpected error: %s", file_path, e)
            raise RuntimeError(f"Error loading {file_path} due to an unexpected error: {e}") from e

    def _load_directory(self, directory: Path) -> Iterator[Document]:
        file_extensions = ['xml', 'md', 'txt', 'html', 'ppt', 'ppx','pdf']
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = Path(root) / file
                    if file.endswith('pdf'):
                        output_dir = Path("experimental_notebook/markdown")
                        output_file = output_dir / file_path.stem / file.replace('.pdf', '.md')
                        command = [
                            "marker_single",
                            str(file_path),
                            str(output_dir),
                            "--batch_multiplier", "2",
                            "--max_pages", "10",
                            "--langs", "English"
                        ]
                        try:
                            result = subprocess.run(command, check=True, capture_output=True, text=True)
                            logger.info("Processed %s to %s", file_path, output_file)
                            file_path = output_file
                        except subprocess.CalledProcessError as e:
                            logger.error("An error occurred while processing %s:", file_path)
                            logger.error(e.stderr)
                            continue
                        except FileNotFoundError:
                            logger.error("The 'marker_single' command was not found. Make sure it's installed and in your PATH.")
                            continue

                    if file_path.is_file():
                        yield from self._load_file(file_path)

    async def _aload_directory(self, directory: Path) -> AsyncIterator[Document]:
        file_extensions = ['xml', 'md', 'txt', 'html', 'ppt', 'ppx', 'pdf']
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = Path(root) / file
                    if file.endswith('pdf'):
                        output_dir = Path("experimental_notebook/markdown")
                        output_file = output_dir / file_path.stem / file.replace('.pdf', '.md')
                        command = [
                            "marker_single",
                            str(file_path),
                            str(output_dir),
                            "--batch_multiplier", "2",
                            "--max_pages", "10",
                            "--langs", "English"
                        ]
                        try:
                            result = subprocess.run(command, check=True, capture_output=True, text=True)
                            logger.info("Processed %s to %s", file_path, output_file)
                            file_path = output_file
                        except subprocess.CalledProcessError as e:
                            logger.error("An error occurred while processing %s:", file_path)
                            logger.error(e.stderr)
                            continue
                        except FileNotFoundError:
                            logger.error("The 'marker_single' command was not found. Make sure it's installed and in your PATH.")
                            continue

                    if file_path.is_file():
                        async for document in self._aload_file(file_path):
                            yield document

    def lazy_load(self) -> Iterator[Document]:
        try:
            for file_path in self.file_paths:
                if file_path.is_dir():
                    yield from self._load_directory(file_path)
                elif file_path.is_file():
                    yield from self._load_file(file_path)
                else:
                    logger.error("The path %s does not exist or is neither a directory nor a file.", file_path)
                    raise ValueError(f"The path {file_path} does not exist or is neither a directory nor a file.")
        except Exception as e:
            logger.error("Error loading file or directory: %s", e)
            raise RuntimeError(f"Error loading file or directory: {e}")

    async def lazy_aload(self) -> AsyncIterator[Document]:
        try:
            for file_path in self.file_paths:
                if file_path.is_dir():
                    async for document in self._aload_directory(file_path):
                        yield document
                elif file_path.is_file():
                    async for document in self._aload_file(file_path):
                        yield document
                else:
                    logger.error("The path %s does not exist or is neither a directory nor a file.", file_path)
                    raise ValueError(f"The path {file_path} does not exist or is neither a directory nor a file.")
        except Exception as e:
            logger.error("Error loading file or directory: %s", e)
            raise RuntimeError(f"Error loading file or directory: {e}")

    def _load_file(self, file_path: Path) -> Iterator[Document]:
        """Load file."""
        elements = self._get_elements(file_path)
        self._post_process_elements(elements)
        if self.mode == "elements":
            for element in elements:
                metadata = self._get_metadata(file_path)
                if hasattr(element, "metadata"):
                    metadata.update(element.metadata.to_dict())
                if hasattr(element, "category"):
                    metadata["category"] = element.category
                yield Document(page_content=str(element), metadata=metadata)
        elif self.mode == "paged":
            text_dict: Dict[int, str] = {}
            meta_dict: Dict[int, Dict] = {}

            for idx, element in enumerate(elements):
                metadata = self._get_metadata(file_path)
                if hasattr(element, "metadata"):
                    metadata.update(element.metadata.to_dict())
                page_number = metadata.get("page_number", 1)

                # Check if this page_number already exists in docs_dict
                if page_number not in text_dict:
                    # If not, create new entry with initial text and metadata
                    text_dict[page_number] = str(element) + "\n\n"
                    meta_dict[page_number] = metadata
                else:
                    # If exists, append to text and update the metadata
                    text_dict[page_number] += str(element) + "\n\n"
                    meta_dict[page_number].update(metadata)

            # Convert the dict to a list of Document objects
            for key in text_dict.keys():
                yield Document(page_content=text_dict[key], metadata=meta_dict[key])
        elif self.mode == "single":
            metadata = self._get_metadata(file_path)
            text = "\n\n".join([str(el) for el in elements])
            yield Document(page_content=text, metadata=metadata)
        else:
            logger.error("Mode %s not supported.", self.mode)
            raise ValueError(f"mode of {self.mode} not supported.")

    async def _aload_file(self, file_path: Path) -> AsyncIterator[Document]:
        elements = await asyncio.to_thread(self._get_elements, file_path)
        self._post_process_elements(elements)
        metadata = await self._aget_metadata(file_path)
        if self.mode == "elements":
            for element in elements:
                element_metadata = metadata.copy()
                if hasattr(element, "metadata"):
                    element_metadata.update(element.metadata.to_dict())
                if hasattr(element, "category"):
                    element_metadata["category"] = element.category
                yield Document(page_content=str(element), metadata=element_metadata)
        elif self.mode == "paged":
            text_dict = {}
            meta_dict = {}
            for idx, element in enumerate(elements):
                element_metadata = metadata.copy()
                if hasattr(element, "metadata"):
                    element_metadata.update(element.metadata.to_dict())
                page_number = element_metadata.get("page_number", 1)

                if page_number not in text_dict:
                    text_dict[page_number] = str(element) + "\n\n"
                    meta_dict[page_number] = element_metadata
                else:
                    text_dict[page_number] += str(element) + "\n\n"
                    meta_dict[page_number].update(element_metadata)

            for key in text_dict.keys():
                yield Document(page_content=text_dict[key], metadata=meta_dict[key])
        elif self.mode == "single":
            text = "\n\n".join([str(el) for el in elements])
            yield Document(page_content=text, metadata=metadata)
        else:
            logger.error("Mode %s not supported.", self.mode)
            raise ValueError(f"mode of {self.mode} not supported.")
