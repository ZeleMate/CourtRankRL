"""
Module for loading legal documents from various text file formats.
"""

import os
import json
import glob
import csv
from typing import List, Dict, Optional, Union
import logging
import concurrent.futures
from functools import lru_cache
import pandas as pd

# Lazy imports to improve startup time
def _import_docx():
    try:
        from docx import Document
        return Document, True
    except ImportError:
        return None, False

def _import_rtf():
    try:
        from striprtf import rtf_to_text
        return rtf_to_text, True
    except ImportError:
        return None, False

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Module-level cache to avoid importing multiple times
_MODULE_CACHE = {
    'docx': None,
    'rtf': None,
}

@lru_cache(maxsize=10)
def get_supported_extensions() -> Dict[str, bool]:
    """Return dictionary of supported extensions and their availability."""
    if not _MODULE_CACHE['rtf']:
        _MODULE_CACHE['rtf'], rtf_available = _import_rtf()
    else:
        rtf_available = _MODULE_CACHE['rtf'] is not None
    
    if not _MODULE_CACHE['docx']:
        _MODULE_CACHE['docx'], docx_available = _import_docx()
    else:
        docx_available = _MODULE_CACHE['docx'] is not None
    
    return {
        '.txt': True,
        '.md': True,
        '.html': True,
        '.csv': True,
        '.json': True,
        '.rtf': rtf_available,
        '.docx': docx_available,
        '.doc': docx_available,  # Support .doc via same processor as .docx
    }

def load_documents_from_folder(folder_path: str, 
                               extensions: Optional[List[str]] = None,
                               encoding: str = 'utf-8',
                               max_workers: int = 4,
                               batch_size: int = 100,
                               include_metadata: bool = False) -> Union[List[str], List[Dict]]:
    """
    Load legal documents from all text files in a specified folder.
    
    Args:
        folder_path: Path to folder containing legal documents
        extensions: List of file extensions to consider (None for all supported)
        encoding: Text encoding to use when reading files
        max_workers: Maximum number of parallel workers for loading files
        batch_size: Process files in batches of this size to limit memory usage
        include_metadata: Whether to include metadata with documents
        
    Returns:
        List of document strings or dictionaries with text and metadata
    """
    if not os.path.exists(folder_path):
        logger.warning(f"Folder not found: {folder_path}")
        return []
    
    # Get supported file extensions
    supported_exts = get_supported_extensions()
    if extensions is None:
        extensions = [ext for ext, available in supported_exts.items() if available]
    else:
        # Filter out unsupported extensions
        extensions = [ext for ext in extensions if ext.lower() in supported_exts and supported_exts[ext.lower()]]
    
    if not extensions:
        logger.warning("No supported file extensions specified")
        return []
    
    logger.info(f"Loading documents from {folder_path} with extensions: {extensions}")
    
    # Find all matching files
    all_files = []
    for ext in extensions:
        file_pattern = os.path.join(folder_path, f"*{ext}")
        all_files.extend(glob.glob(file_pattern))
    
    if not all_files:
        logger.warning(f"No matching files found in {folder_path}")
        return []
    
    logger.info(f"Found {len(all_files)} files to process")
    
    # Process files in batches to limit memory usage
    documents = []
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        batch_docs = _process_files_batch(batch_files, encoding, max_workers, include_metadata)
        documents.extend(batch_docs)
        logger.info(f"Processed batch {i//batch_size + 1}: {len(batch_docs)} documents loaded")
    
    logger.info(f"Loaded {len(documents)} documents total")
    return documents

def _process_files_batch(file_paths: List[str], 
                         encoding: str,
                         max_workers: int,
                         include_metadata: bool) -> List[Union[str, Dict]]:
    """Process a batch of files using parallel execution."""
    results = []
    
    # Use ThreadPoolExecutor for I/O bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all file loading tasks
        future_to_file = {
            executor.submit(_load_document_from_file, file_path, encoding, include_metadata): file_path 
            for file_path in file_paths
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
    
    return results

def _load_document_from_file(file_path: str, encoding: str, include_metadata: bool) -> Optional[Union[str, Dict]]:
    """
    Load a single document from a text file with optional metadata.
    
    Args:
        file_path: Path to the document file
        encoding: Text encoding to use
        include_metadata: Whether to include metadata
        
    Returns:
        Document content as string/dict or None on error
    """
    try:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Basic metadata that's available for all files
        metadata = {
            'filename': os.path.basename(file_path),
            'file_path': file_path,
            'file_type': ext[1:],  # Remove the dot
            'file_size': os.path.getsize(file_path)
        }
        
        content = None
        
        # Handle RTF files
        if ext == '.rtf':
            if not _MODULE_CACHE['rtf']:
                _MODULE_CACHE['rtf'], _ = _import_rtf()
            
            if _MODULE_CACHE['rtf']:
                with open(file_path, 'r', encoding=encoding, errors="ignore") as file:
                    raw_content = file.read()
                    content = _MODULE_CACHE['rtf'](raw_content)
            else:
                logger.warning("striprtf not installed. Skipping RTF file.")
                return None
        
        # Handle DOCX files
        elif ext in ['.docx', '.doc']:
            if not _MODULE_CACHE['docx']:
                _MODULE_CACHE['docx'], _ = _import_docx()
            
            if _MODULE_CACHE['docx']:
                doc = _MODULE_CACHE['docx'](file_path)
                content = "\n".join(p.text for p in doc.paragraphs)
                
                # Extract additional metadata if available
                if hasattr(doc, 'core_properties'):
                    if doc.core_properties.author:
                        metadata['author'] = doc.core_properties.author
                    if doc.core_properties.created:
                        metadata['created'] = doc.core_properties.created
                    if doc.core_properties.modified:
                        metadata['modified'] = doc.core_properties.modified
                    if doc.core_properties.title:
                        metadata['title'] = doc.core_properties.title
            else:
                logger.warning("python-docx not installed. Skipping DOCX file.")
                return None
            
        # Handle JSON files specially
        elif ext == '.json':
            with open(file_path, 'r', encoding=encoding) as file:
                data = json.load(file)
                
                if isinstance(data, str):
                    content = data
                elif isinstance(data, dict):
                    # Extract text content
                    content = data.get('text', data.get('content', ''))
                    
                    # Extract additional metadata
                    for key, value in data.items():
                        if key not in ['text', 'content'] and isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                else:
                    content = json.dumps(data, ensure_ascii=False)
        
        # Handle CSV files
        elif ext == '.csv':
            text_content = []
            with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    text_content.append(", ".join(row))
            content = "\n".join(text_content)

        # Handle plain text files
        else:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                content = file.read().strip()
        
        if include_metadata:
            return {
                'text': content,
                'metadata': metadata
            }
        return content
                
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None

def load_documents_with_metadata(folder_path: str, 
                                extensions: Optional[List[str]] = None,
                                encoding: str = 'utf-8',
                                max_workers: int = 4) -> List[Dict]:
    """
    Load documents with additional metadata.
    
    Args:
        folder_path: Path to folder containing legal documents
        extensions: List of file extensions to consider
        encoding: Text encoding to use
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of dictionaries with document text and metadata
    """
    return load_documents_from_folder(folder_path, extensions, encoding, 
                                     max_workers, include_metadata=True)

def is_format_supported(extension: str) -> bool:
    """Check if a file format is supported by the loader."""
    supported = get_supported_extensions()
    return extension.lower() in supported and supported[extension.lower()]

def load_processed_documents_from_csv(file_path: str, text_column: str = 'text') -> List[str]:
    """
    Load document text from a preprocessed CSV file (potentially gzipped).

    Args:
        file_path: Path to the CSV file (e.g., .csv or .csv.gz).
        text_column: The name of the column containing the document text.

    Returns:
        List of document strings.
    """
    logger.info(f"Loading processed documents from {file_path}")
    try:
        # Pandas automatically handles .gz compression based on extension
        df = pd.read_csv(file_path)
        if text_column not in df.columns:
            logger.error(f"Text column '{text_column}' not found in {file_path}. Available columns: {df.columns.tolist()}")
            return []
        
        # Drop rows where the text column is NaN or empty, convert to string
        documents = df[text_column].dropna().astype(str).tolist()
        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    except FileNotFoundError:
        logger.error(f"Processed data file not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading processed data from {file_path}: {e}")
        return []