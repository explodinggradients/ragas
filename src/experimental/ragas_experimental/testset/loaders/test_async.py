from ragas_loader import RAGASLoader
import traceback
import asyncio
async def main():
    loader = RAGASLoader(file_path="./path", mode="single", autodetect_encoding=False)
    try:
        docs = await loader.aload()
        print(len(docs))
        for i, doc in enumerate(docs):
            file_name = doc.metadata["source"].split("/")[-1]

            print(f"\n{'=' * 80}")
            print(f"File {i}: {file_name}")
            print(f"{'=' * 80}")

            print("\nParsed Content:")
            print("-" * 40)
            print(doc.page_content)

            print("\nRaw Content (first 1000 characters):")
            print("-" * 40)
            print(doc.metadata["raw_content"][:5000])

            print(f"\n{'=' * 80}\n")
    except Exception as e:
        print(f"Error in main execution: {e}")
        print(traceback.format_exc())


# Test Async Loading
if __name__ == "__main__":
    asyncio.run(main())
