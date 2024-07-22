from ragas_loader import RAGASLoader
import traceback
def main():
    loader = RAGASLoader(file_path="./experimental_notebook", mode="single", autodetect_encoding=False)
    try:
        docs = list(loader.lazy_load())  # Use lazy_load directly and convert to list
        print(len(docs))
        for i, doc in enumerate(docs):
            file_name = doc.metadata["source"].split("/")[-1]

            print(f"\n{'=' * 80}")
            print(f"File {i}: {file_name}")
            print(f"{'=' * 80}")

            print("\nParsed Content (first 1000 characters):")
            print("-" * 40)
            print(doc.page_content[:1000])

            print("\nRaw Content (first 1000 characters):")
            print("-" * 40)
            print(doc.metadata["raw_content"][:1000])

            print(f"\n{'=' * 80}\n")
    except Exception as e:
        print(f"Error in main execution: {e}")
        print(traceback.format_exc())

if __name__=="__main__":
    main()