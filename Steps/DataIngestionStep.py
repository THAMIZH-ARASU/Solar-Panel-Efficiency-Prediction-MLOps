import pandas as pd
import os

from Src.IngestData import DataIngestorFactory
# from zenml import step


# @step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """Ingest data from a ZIP file using the appropriate DataIngestor."""
    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1]  # Since we're dealing with ZIP files, this is hardcoded

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest(file_path)
    return df
