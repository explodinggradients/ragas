from uuid import UUID

import pandas as pd
import pytest
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from langsmith import Client

client = Client()


# datarows with queries and ground truth
# sample questions
questions = [
    "What are the advantages of remote work? Why does gitlab do it?",
    "what are the dis-advantages of remote work in gitlab? How do you mitigate that?",
    "What does it mean to be 'all-remote'? Why is it important?",
    "How does compensation work in an 'all-remote' setting?",
    "How to run effective meetings in 'all-remote' setting",
]

answers = [
    "Remote work offers numerous advantages including increased flexibility, the ability to hire top talent globally without geographical constraints, enhanced productivity due to fewer distractions, and significant cost savings on office space and related expenses. GitLab adopts an all-remote model to leverage these benefits, ensuring a more inclusive and diverse workforce, fostering a culture that values output over presence, and providing employees the freedom to work in environments that best suit their personal and professional needs. This approach not only supports individual employee well-being and work-life balance but also positions GitLab as a resilient and adaptive organization in a rapidly changing global work landscape.",
    "GitLab's remote work challenges include onboarding difficulties, loneliness, communication breakdowns, work/life balance issues, time zone conflicts, and the need for strong time management skills. To mitigate these, GitLab employs strategies such as providing comprehensive onboarding resources, fostering community through virtual coffee chats and social calls, prioritizing asynchronous communication, reimbursing coworking spaces, empowering employees to manage their own schedules, focusing on results rather than hours, and screening for self-discipline during the hiring process. These measures aim to enhance productivity and employee satisfaction in a remote work setting.",
    "Being 'all-remote' means that an organization empowers every individual to work from any location where they feel most fulfilled, without the need to report to a company-owned office, thereby treating all employees equally regardless of their physical location. This approach is important because it eliminates location hierarchy, allowing for a more inclusive work environment where team members have the autonomy to create their ideal workspace and can maintain their job regardless of life changes, such as relocations due to family commitments. It supports a diverse workforce, including caregivers, working parents, and military spouses, by providing them with the flexibility to work from anywhere, fostering equality among all employees and enabling a global talent pool without the constraints of geographical boundaries.",
    "In an 'all-remote' setting, such as at GitLab, compensation is structured around local rates rather than a single global standard, which means employees are paid based on the cost of living and market rates in their respective locations. This approach allows the company to hire globally without being bound by the high salary standards of any particular region, like San Francisco. GitLab uses a compensation calculator to ensure transparency and fairness in pay, adjusting salaries based on a combination of factors including location, experience, and market data. Payments are typically made in the local currency of the employee, and for countries where direct employment isn't feasible, GitLab utilizes professional employment organizations or hires contractors. This model supports GitLab's global talent acquisition strategy while managing compensation costs effectively.",
    "To run effective meetings in an 'all-remote' setting, it's crucial to be intentional about meeting necessity, provide clear agendas and supporting materials in advance, start and end on time, document discussions in real time, and make attendance optional to respect time zones and individual schedules. Recording meetings for asynchronous viewing, using reliable communication tools like Zoom, and ensuring active participation through video feedback are also key practices. This approach aligns with GitLab's guidelines for maximizing efficiency, inclusivity, and collaboration in a remote work environment.",
]

dataset = {"question": questions, "ground_truth": answers}


def upload_to_langsmith(dataset_name: str) -> UUID:
    # Creating a pandas DataFrame from the dataset dictionary
    df = pd.DataFrame(dataset)

    # upload to langsmith
    langsmith_dataset = client.upload_dataframe(
        name=dataset_name,
        description="temporal dataset for testing langsmith",
        df=df,
        input_keys=["question"],
        output_keys=["ground_truth"],
    )

    return langsmith_dataset.id


def clean_langsmith(langsmith_dataset_id: UUID):
    # clean langsmith
    client.delete_dataset(dataset_id=langsmith_dataset_id)


def llm_chain_factory() -> Runnable:
    # just LLM
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    Question: {question}

    Helpful Answer:"""
    llm_prompt = PromptTemplate.from_template(template)

    # LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # just llm pipeline
    just_llm = (
        {"question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
        | RunnableParallel(
            {
                "answer": RunnablePassthrough(),
                "contexts": RunnableLambda(lambda _: [""]),
            }
        )
    )

    return just_llm


@pytest.fixture()
def langsmith_dataset():
    dataset_name = "temporal_dataset"
    langsmith_dataset_id = upload_to_langsmith(dataset_name)
    yield dataset_name
    clean_langsmith(langsmith_dataset_id)


@pytest.mark.e2e()
def test_langsmith_evaluate(langsmith_dataset):
    # setup
    just_llm = llm_chain_factory()

    from ragas.integrations.langsmith import evaluate
    from ragas.metrics import answer_correctness

    # evaluate just llms
    _ = evaluate(
        dataset_name=langsmith_dataset,
        llm_or_chain_factory=just_llm,
        # experiment_name="just_llm",
        metrics=[answer_correctness],
        verbose=True,
    )
