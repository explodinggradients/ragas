from langchain.prompts import AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate, AIMessagePromptTemplate

# Context Recall

CONTEXT_RECALL_HUMAN: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
    """<instructions>
Given a context, and an answer, first identify each individual sentence in the answer, then analyze each, and classify if the sentence can be attributed to the given context or not.
Carefully think step by step and provide detailed reasoning reflecting the context provided before coming to conclusion. 
Follow the exact output format as shown in the example response. NEVER output double newlines in the response!
</instructions>

Here are some examples with responses:

<example_input>
<context>
Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist,widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.
</context>

<answer>
Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. He published 4 papers in 1905.  Einstein moved to Switzerland in 1895
</answer>
</example_input>

<example_response>
Line by line sentence classifications for the given answer:
1. Albert Einstein born in 14 March 1879 was  German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. The date of birth of Einstein is mentioned clearly in the context. So [Attributed]
2. He received the 1921 Nobel Prize in Physics "for his services to theoretical physics. The exact sentence is present in the given context. So [Attributed]
3. He published 4 papers in 1905. There is no mention about papers he wrote in given the context. So [Not Attributed]
4. Einstein moved to Switzerland in 1895. There is not supporting evidence for this in the given the context. So [Not Attributed]
</example_response>

<example_input>
<context>
William Shakespeare (26 April 1564 – 23 April 1616) was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language and the world's greatest dramatist. He is often called England's national poet and the "Bard of Avon". His extant works, including some collaborations, consist of about 39 plays, 154 sonnets, and two long narrative poems. His plays have been translated into every major living language and are performed more often than those of any other playwright. He was born in Stratford-upon-Avon and married Anne Hathaway, with whom he had three children.
</context>

<answer>
William Shakespeare was born on 26 April 1564 in Stratford-upon-Avon. He wrote 39 plays and 154 sonnets. Shakespeare was known as the Bard of Avon. He had two children with Anne Hathaway.
</answer>
</example_input>

<example_response>
Line by line sentence classifications for the given answer:
1. William Shakespeare was born on 26 April 1564 in Stratford-upon-Avon. The date and place of birth are mentioned clearly in the context. So [Attributed]
2. He wrote 39 plays and 154 sonnets. The exact number of plays and sonnets is present in the given context. So [Attributed]
3. Shakespeare was known as the Bard of Avon. This title is specifically mentioned in the context. So [Attributed]
4. He had two children with Anne Hathaway. The context states that he had three children with Anne Hathaway, not two. So [Not Attributed]
</example_response>

Here is the input and answer for you to classify:

<context>
{context}
</context>

<answer>
{ground_truth}
</answer>

Remember it is absolutely critical that you do not output double newlines in the response!
"""  # noqa: E501
)

CONTEXT_RECALL_AI: AIMessagePromptTemplate = AIMessagePromptTemplate.from_template(
    """Line by line sentence classifications for the given answer:
"""
)

# Context Precision

CONTEXT_PRECISION_HUMAN: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
    """<instructions>
Given a question and a context, verify if the information in the given context is directly relevant for answering the question.
Answer only with a single word of either "Yes" or "No" and nothing else.
</instructions>

<example_input>
<question>
What is the significance of the Statue of Liberty in New York City?
</question>

<context>
The Statue of Liberty National Monument and Ellis Island Immigration Museum are managed by the National Park Service and are in both New York and New Jersey. They are joined in the harbor by Governors Island National Monument. Historic sites under federal management on Manhattan Island include Stonewall National Monument; Castle Clinton National Monument; Federal Hall National Memorial; Theodore Roosevelt Birthplace National Historic Site; General Grant National Memorial (Grant's Tomb); African Burial Ground National Monument; and Hamilton Grange National Memorial. Hundreds of properties are listed on the National Register of Historic Places or as a National Historic Landmark.
</context>
</example_input>

<example_response>Yes</example_response>

Here is the question and context for you to analyze:
<question>
{question}
</question>

<context>
{context}
</context>

Remember, you must answer with a single word of either "Yes" or "No" and nothing else!
"""  # noqa: E501
)

CONTEXT_PRECISION_AI: AIMessagePromptTemplate = AIMessagePromptTemplate.from_template(
    """The single word answer is: """
)

# Answer Relevancy

ANSWER_RELEVANCY_HUMAN: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
    """<instructions>
Generate a question for the given answer. Follow the exact output format as shown in the example response. Do not add anything extra in the response!
</instructions>

<example_input>
<answer>The PSLV-C56 mission is scheduled to be launched on Sunday, 30 July 2023 at 06:30 IST / 01:00 UTC. It will be launched from the Satish Dhawan Space Centre, Sriharikota, Andhra Pradesh, India</answer>
</example_input>

<example_response>
Question: When is the scheduled launch date and time for the PSLV-C56 mission, and where will it be launched from?
</example_response>

Here is the input:
<answer>{answer}</answer>
"""  # noqa: E501
)

ANSWER_RELEVANCY_AI: AIMessagePromptTemplate = AIMessagePromptTemplate.from_template(
    """Question: """
)

# Faithfulness

FAITHFULNESS_STATEMENTS_HUMAN: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
    """<instructions>
Given a question and answer, create one or more statements from each sentence in the given answer. Each sentence should be a standalone statement and includes a subject.
Follow the exact output format as shown in the example responses. Notice that there should not be any blank lines, tags, or numbering in the response!
</instructions>

<example_input>
<question>
Who was  Albert Einstein and what is he best known for?
</question>

<answer>
He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
</answer>
</example_input>

<example_response>
statements:
Albert Einstein was born in Germany.
Albert Einstein was best known for his theory of relativity.
</example_response>

<example_input>
<question>
Cadmium Chloride is slightly soluble in this chemical, it is also called what?
</question>

<answer>
alcohol
</answer>
</example_input>

<example_response>
statements:
Cadmium Chloride is slightly soluble in alcohol.
</example_response>

<example_input>
<question>
Were Shahul and Jithin of the same nationality?
</question>

<answer>
They were from different countries.
</answer>
</example_input>

<example_response>
statements:
Shahul and Jithin were from different countries.
</example_response>

Now here is the question and answer for you to create statements from:
<question>
{question}
</question>

<answer>
{answer}
</answer>

Remember, it's very important that you follow the instructions and output format exactly!
"""  # noqa: E501
)

FAITHFULNESS_STATEMENTS_AI: AIMessagePromptTemplate = AIMessagePromptTemplate.from_template(
    """Here is the response for the above question and answer without any blank lines:
statements:
"""
)

FAITHFULNESS_VERDICTS_HUMAN: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
    """<instructions>
Consider the given context and following statements, then determine whether they are supported by the information present in the context. Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. 
Follow the exact output format as shown in the below example. Importantly, NEVER use two consecutive newlines in your response.
</instructions>

<example_input>
<context>
John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
</context>

<statements>
1. John is majoring in Biology.\n2. John is taking a course on Artificial Intelligence.\n3. John is a dedicated student.\n4. John has a part-time job.\n5. John is interested in computer programming.
</statements>
</example_input>

<example_response>
Answer:
1. John is majoring in Biology.
Explanation: John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.  Verdict: No.
2. John is taking a course on Artificial Intelligence.
Explanation: The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI. Verdict: No.
3. John is a dedicated student.
Explanation: The prompt states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication. Verdict: Yes.
4. John has a part-time job.
Explanation: There is no information given in the context about John having a part-time job. Therefore, it cannot be deduced that John has a part-time job.  Verdict: No.
5. John is interested in computer programming.
Explanation: The context states that John is pursuing a degree in Computer Science, which implies an interest in computer programming. Verdict: Yes.
Final verdict for each statement in order: No. No. Yes. No. Yes.
</example_response>

Now here is the context and statements for you to classify:

<context>
{context}
</context>

<statements>
{statements}
</statements>

Remember, it's very important that you do not output double newlines in the response. Each line should contain a statement, explanation, and verdict!
"""  # noqa: E501
)

FAITHFULNESS_VERDICTS_AI: AIMessagePromptTemplate = AIMessagePromptTemplate.from_template(
    """Here is the answer in the exact example_response format without any blank lines:
Answer:
"""
)

# Aspect Critique

ASPECT_CRITIQUE_HUMAN: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
    """<instructions>
Given an input and submission, evaluate the submission only using the given criteria.
Think step by step providing reasoning and arrive at a conclusion at the end by generating a single word 'Yes' or 'No' verdict at the end.
</instructions>

<example_input>
<input>Who was the director of Los Alamos Laboratory?</input>

<submission>Einstein was the director of  Los Alamos Laboratory.</submission>

<criteria>Is the output written in perfect grammar?</criteria>
</example_input>

<example_response>Here are my thoughts: the criteria for evaluation is whether the output is written in perfect grammar. In this case, the output is grammatically correct. Therefore, the answer is:

Yes</example_response>

<example_input>
<input>What is the capital of Italy?</input>

<submission>Rome is capital of Italy.</submission>

<criteria>Is the output written in perfect grammar</criteria>
</example_input>

<example_response>Evaluating this submission against the specified criteria of perfect grammar, we observe that the sentence "Rome is capital of Italy." lacks the necessary article 'the' before 'capital.' The grammatically correct form should be "Rome is the capital of Italy." Therefore, according to the criteria of perfect grammar, the verdict is:

No</example_response>

Now here is the input, submission and criteria for you to evaluate:

<input>{input}</input>

<submission>{submission}</submission>

<criteria>{criteria}</criteria>

Remember, it's critical that your verdict is a single word of either 'Yes' or 'No' and nothing else!
"""  # noqa: E501
)

ASPECT_CRITIQUE_AI: AIMessagePromptTemplate = AIMessagePromptTemplate.from_template(
    """Here are my thoughts followed by a single word "yes" or "no" verdict:
"""
)

ASPECT_GT_CRITIQUE_HUMAN: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
    """<instructions>
Given a question, generated answer, and ground truth answer, evaluate the generated answer using the given criteria.
Think step by step, providing reasoning, and conclude with a single word 'Yes' or 'No' verdict.
</instructions>

<example_input>
<question>What are the functions of Vitamin C in the human body? Answer using context: Vitamin C is commonly found in citrus fruits and is essential for health.</question>
<ground_truth_answer>Vitamin C plays various roles, including immune support and collagen synthesis, but further details are not provided in the context.</ground_truth_answer>
<generated_answer>Vitamin C is crucial for immune system function and helps prevent scurvy.</generated_answer>
<criteria>Did the generated answer correctly assess the sufficiency of the context to provide an accurate response?</criteria>
</example_input>

<example_response>The context mentions that Vitamin C is essential for health and found in citrus fruits, but it does not specify its functions. The generated answer states that Vitamin C supports the immune system and prevents scurvy. While these statements are true, they are not corroborated by the provided context. Therefore, the model did not correctly assess the context's sufficiency. The answer is:

No</example_response>

<example_input>
<question>Explain the role of enzyme X in the human digestive system. Answer using context: Enzyme X's function is currently under research.</question>
<ground_truth_answer>Enzyme X's specific role is not fully understood yet.</ground_truth_answer>
<generated_answer>There is insufficient context information to detail the exact role of enzyme X in the human digestive system.</generated_answer>
<criteria>Did the generated answer correctly assess the sufficiency of the context to provide an accurate response?</criteria>
</example_input>

<example_response>The ground truth indicates that the role of enzyme X is not fully understood, making the context insufficient for a detailed explanation. The generated answer correctly identifies this insufficiency. Hence, the answer is:

Yes</example_response>

<example_input>
<question>What are the primary colors in traditional color theory? Answer using context: Traditional color theory states that red, blue, and yellow are the primary colors, which cannot be created by mixing other colors.</question>
<ground_truth_answer>The primary colors in traditional color theory are red, blue, and yellow.</ground_truth_answer>
<generated_answer>Red, blue, and yellow are the primary colors in traditional color theory.</generated_answer>
<criteria>Did the generated answer correctly assess the sufficiency of the context to provide an accurate response?</criteria>
</example_input>

<example_response>The question asks for the primary colors in traditional color theory, and the provided context clearly states that these are red, blue, and yellow. The generated answer directly matches this information from the context, correctly identifying the primary colors as red, blue, and yellow. Given the context's sufficiency and the generated answer's accuracy, the answer is:

Yes</example_response>

<example_input>
<question>Who discovered penicillin? Answer using context: Penicillin was a groundbreaking discovery in medical science.</question>
<ground_truth_answer>Alexander Fleming discovered penicillin.</ground_truth_answer>
<generated_answer>The discovery of penicillin was made by Alexander Fleming.</generated_answer>
<criteria>Does the generated answer accurately match the ground truth answer?</criteria>
</example_input>

<example_response>Considering the question along with its context about penicillin's significance in medical science, the focus is on the discoverer of penicillin. The generated answer aligns with the ground truth, correctly identifying Alexander Fleming as the discoverer. Therefore, the answer is:

Yes</example_response>

Now here is the input, submission, ground truth, and criteria for you to evaluate:

<question>{input}</question>
<ground_truth_answer>{ground_truth}</ground_truth_answer>
<generated_answer>{submission}</generated_answer>
<criteria>{criteria}</criteria>

Remember, it's critical that your verdict is a single word of either 'Yes' or 'No' and nothing else!
"""  # noqa: E501
)
