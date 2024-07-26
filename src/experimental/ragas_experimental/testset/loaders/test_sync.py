import logging
import traceback
from ragas_loader import RAGASLoader

# Setup logging
logger = logging.getLogger(__name__)

def main():
    loader = RAGASLoader(file_path="./path", mode="single", autodetect_encoding=False)
    try:
        docs = loader.load()
        logger.info("Number of documents loaded: %d", len(docs))
        for i, doc in enumerate(docs):
            file_name = doc.metadata["source"].split("/")[-1]

            logger.info("\n%s", '=' * 80)
            logger.info("File %d: %s", i, file_name)
            logger.info("%s", '=' * 80)

            logger.info("\nParsed Content (first 1000 characters):")
            logger.info("%s", '-' * 40)
            logger.info("%s", doc.page_content[:1000])

            logger.info("\nRaw Content (first 1000 characters):")
            logger.info("%s", '-' * 40)
            logger.info("%s", doc.metadata["raw_content"][:1000])

            logger.info("\n%s\n", '=' * 80)
    except Exception as e:
        logger.error("Error in main execution: %s", e)
        logger.error("%s", traceback.format_exc())

if __name__ == "__main__":
    main()
