import json
import typing as t

from ragas.messages import AIMessage, HumanMessage


def get_last_orchestration_value(traces: t.List[t.Dict[str, t.Any]], key: str):
    """
    Iterates through the traces to find the last occurrence of a specified key
    within the orchestrationTrace.

    Returns:
        (index, value): Tuple where index is the last index at which the key was found, and value is the corresponding value, or (None, None) if not found.
    """
    last_index = -1
    last_value = None
    for i, trace in enumerate(traces):
        orchestration = trace.get("trace", {}).get("orchestrationTrace", {})
        if key in orchestration:
            last_index = i
            last_value = orchestration[key]
    return last_index, last_value


def extract_messages_from_model_invocation(model_inv):
    """
    Extracts messages from the 'text' field of the modelInvocationInput.
    Ensures that each message's content is cast to a string.

    Returns:
        List of messages as HumanMessage or AIMessage objects.
    """
    messages = []
    text_json = json.loads(model_inv.get("text", "{}"))
    for msg in text_json.get("messages", []):
        content_str = str(msg.get("content", ""))
        role = msg.get("role")
        if role == "user":
            messages.append(HumanMessage(content=content_str))
        elif role == "assistant":
            messages.append(AIMessage(content=content_str))
    return messages[:-1]


def convert_to_ragas_messages(traces: t.List):
    """
    Converts a list of trace dictionaries into a list of messages.
    It extracts messages from the last modelInvocationInput and appends
    the finalResponse from the observation (if it occurs after the model invocation).

    Returns:
        List of HumanMessage and AIMessage objects.
    """
    result = []

    # Get the last modelInvocationInput from the traces.
    last_model_inv_index, last_model_inv = get_last_orchestration_value(
        traces, "modelInvocationInput"
    )
    if last_model_inv is not None:
        result.extend(extract_messages_from_model_invocation(last_model_inv))

    # Get the last observation from the traces.
    last_obs_index, last_observation = get_last_orchestration_value(
        traces, "observation"
    )
    if last_observation is not None and last_obs_index > last_model_inv_index:
        final_text = str(last_observation.get("finalResponse", {}).get("text", ""))
        result.append(AIMessage(content=final_text))

    return result


def extract_kb_trace(traces):
    """
    Extracts groups of traces that follow the specific order:
      1. An element with 'trace' -> 'orchestrationTrace' containing an 'invocationInput'
         with invocationType == "KNOWLEDGE_BASE"
      2. Followed (later in the list or within the same trace) by an element with an 'observation'
         that contains 'knowledgeBaseLookupOutput'
      3. Followed by an element with an 'observation' that contains 'finalResponse'

    Returns a list of dictionaries each with keys:
      'user_input', 'retrieved_contexts', and 'response'

    This version supports multiple knowledge base invocation groups.
    """
    results = []
    groups_in_progress = []  # list to keep track of groups in progress

    for trace in traces:
        orchestration = trace.get("trace", {}).get("orchestrationTrace", {})

        # 1. Look for a KB invocation input.
        inv_input = orchestration.get("invocationInput")
        if inv_input and inv_input.get("invocationType") == "KNOWLEDGE_BASE":
            kb_input = inv_input.get("knowledgeBaseLookupInput", {})
            # Start a new group with the user's input text.
            groups_in_progress.append({"user_input": kb_input.get("text")})

        # 2. Process observations.
        obs = orchestration.get("observation", {})
        if obs:
            # If the observation contains a KB output, assign it to the earliest group
            # that does not yet have a 'retrieved_contexts' key.
            if "knowledgeBaseLookupOutput" in obs:
                for group in groups_in_progress:
                    if "user_input" in group and "retrieved_contexts" not in group:
                        kb_output = obs["knowledgeBaseLookupOutput"]
                        group["retrieved_contexts"] = [
                            retrieved.get("content", {}).get("text")
                            for retrieved in kb_output.get("retrievedReferences", [])
                        ]
                        break

            # 3. When we see a final response, assign it to all groups that have already
            # received their KB output but still lack a response.
            if "finalResponse" in obs:
                final_text = obs["finalResponse"].get("text")
                completed_groups = []
                for group in groups_in_progress:
                    if (
                        "user_input" in group
                        and "retrieved_contexts" in group
                        and "response" not in group
                    ):
                        group["response"] = final_text
                        completed_groups.append(group)
                # Remove completed groups from the in-progress list and add to the final results.
                groups_in_progress = [
                    g for g in groups_in_progress if g not in completed_groups
                ]
                results.extend(completed_groups)

    return results
