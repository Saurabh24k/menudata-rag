import os
import gdown
import logging

logger = logging.getLogger(__name__)

def download_file_from_google_drive(link_or_id, output_path):
    
    if os.path.exists(output_path):
        logger.info(f"File already exists: {output_path}")
        return
    logger.info(f"Downloading {link_or_id} to {output_path} ...")
    gdown.download(link_or_id, output_path, quiet=False)
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Download failed for {link_or_id}")
