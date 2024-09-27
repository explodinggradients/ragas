# Testset Generation for RAG

In RAG application, when a user interacts through your application to a set of documents the user may ask different types of queries. These queries in terms of a RAG system can be generally classified into two types:

## Two fundamental query types in RAG

- Specific Queries
    - Queries directly answerable by referring to single context
    - “What is the value of X in Report FY2020 ?”

- Abstract Queries

    - Queries that can only be answered by referring to multiple documents
    - “What is the the revenue trend for Company X from FY2020 through FY2023?”


Synthesizing specific queries is relatively easy as it requires only a single context to generate the query. However, abstract queries require multiple contexts to generate the query.** Now the fundamental question is how select the right set of chunks to generate the abstract queries**. Different types of abstract queries require different types of contexts. For example, 

- Abstract queries comparing two entities in a specific domain require contexts that contain information about the entities.
    - “Compare the revenue growth of Company X and Company Y from FY2020 through FY2023”
- Abstract queries about the a topic discussed in different contexts require contexts that contain information about the topic.
    - “What are the different strategies used by companies to increase revenue?”


To solve this problem, Ragas uses a Knowledge Graph based approach to Testset Generation.

## Knowledge Graph Creation

<div style="text-align: center;">
    <img src="/docs/_static/imgs/kg_rag.png" alt="KG Formation" width="auto" height="auto">
</div>




## Scenario Generation
