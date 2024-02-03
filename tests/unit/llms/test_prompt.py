import importlib
import pkgutil

import ragas
from ragas.llms.prompt import Prompt

TESTCASES = [
    {
        "name": "test-prompt",
        "instruction": "Create one or more statements from each sentence in the given answer.",
        "examples": [
            {
                "question": "Cadmium Chloride is slightly soluble in this chemical, it is also called what?",
                "answer": "alcohol",
                "statements in json": """{
                                                    "statements": [
                                                    "Cadmium Chloride is slightly soluble in alcohol."
                                                    ]
                                                }""",
            },
            {
                "question": "Were Hitler and Benito Mussolini of the same nationality?",
                "answer": "Sorry, I can't provide answer to that question.",
                "statements in json": """{
                                                    "statements": []
                                                }""",
            },
        ],
        "input_keys": ["question", "answer"],
        "output_key": "statements in json",
    },
    {
        "name": "test-prompt",
        "instruction": 'Natural language inference. Use only "Yes" (1) or "No" (0) as a binary verdict.',
        "examples": [
            {
                "Context": """John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.
                                        statement_1: John is majoring in Biology.
                                        statement_2: John is taking a course on Artificial Intelligence. 
                                        statement_3: John is a dedicated student. 
                                        statement_4: John has a part-time job.""",
                "Answer": """[
                                {
                                    "statement_1": "John is majoring in Biology.",
                                    "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                                    "verdict": "0"
                                },
                                {
                                    "statement_2": "John is taking a course on Artificial Intelligence.",
                                    "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                                    "verdict": "0"
                                },
                                {
                                    "statement_3": "John is a dedicated student.",
                                    "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                                    "verdict": "1"
                                },
                                {
                                    "statement_4": "John has a part-time job.",
                                    "reason": "There is no information given in the context about John having a part-time job.",
                                    "verdict": "0"
                                }]
                                """,
            }
        ],
        "input_keys": ["Context"],
        "output_key": "Answer",
        "output_type": "json",
    },
    {
        "name": "test-prompt",
        "instruction": "This is a test prompt without examples",
        "input_keys": ["Context"],
        "output_key": "Answer",
        "output_type": "json",
    },
]


def test_prompt_object():
    for testcase in TESTCASES:
        prompt = Prompt(**testcase)

        assert prompt is not None, "Prompt object is not created"
        assert (
            prompt.instruction == testcase["instruction"]
        ), "instruction in object is not same as in the testcase"
        assert (
            prompt.input_keys == testcase["input_keys"]
        ), "input_keys in object is not same as in the testcase"
        assert (
            prompt.output_key == testcase["output_key"]
        ), "output_key in object is not same as in the testcase"
        assert prompt.output_type == testcase.get(
            "output_type", "json"
        ), "output_type in object is not same as in the testcase"
        assert prompt.examples == testcase.get(
            "examples", []
        ), "examples should be empty if not provided"
        if testcase.get("examples"):
            assert isinstance(
                prompt.get_example_str(0), str
            ), "get_example_str should return a string"


def test_prompt_object_names():
    "ensure that all prompt objects have unique names"

    prompt_object_names = []
    # Iterate through all modules in the ragas package
    for module_info in pkgutil.iter_modules(ragas.__path__):
        module = importlib.import_module(f"ragas.{module_info.name}")

        # Iterate through all objects in the module
        for obj_name in dir(module):
            obj = getattr(module, obj_name)

            # Check if the object is an instance of Prompt
            if isinstance(obj, Prompt):
                assert (
                    obj.name not in prompt_object_names
                ), f"Duplicate prompt name: {obj.name}"
                prompt_object_names.append(obj.name)
