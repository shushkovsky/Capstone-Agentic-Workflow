### NOTE: ###
# This file is generated from the original work in agentic_workflow.ipynb
# It returns the langgraph invoker to be called in a Streamlit app.

def main():
    import os
    import json
    from dotenv import load_dotenv
    from typing import Annotated, List, Literal, Optional
    from typing_extensions import TypedDict
    from pydantic import BaseModel, Field
    from operator import add
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.example_selectors import SemanticSimilarityExampleSelector
    from langchain_neo4j import Neo4jGraph, Neo4jVector
    from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
    from langgraph.graph import END, START, StateGraph
    from neo4j.exceptions import CypherSyntaxError
    from IPython.display import Markdown
    from IPython.display import Image, display
    from langgraph.graph import END, START, StateGraph
    from langchain_openai import ChatOpenAI
    import getpass

    load_dotenv()

    graph_with_schema = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        enhanced_schema=False
    )
    graph_with_schema.refresh_schema()

    #print(graph_with_schema.schema)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    class InputState(TypedDict):
        question: str


    class OverallState(TypedDict):
        question: str
        next_action: str
        cypher_statement: str
        cypher_errors: List[str]
        database_records: List[dict]
        steps: Annotated[List[str], add]


    class OutputState(TypedDict):
        answer: str
        steps: List[str]
        cypher_statement: str

    def reject_question(state: OverallState) -> OutputState:
        return {
            "answer": "Sorry, I can't answer questions unrelated to genes, drugs, or their targets.",
            "cypher_statement": "",
            "steps": state.get("steps", []) + ["rejected"],
        }

    guardrails_system = """
    As an intelligent assistant, your primary objective is to decide whether a given question is related to genes, drugs, and their targets.
    If the question is related to genes, drugs, and their targets, output "drug". Otherwise, output "end".
    To make this decision, assess the content of the question and determine if it refers to any genes, drugs, targets,
    or related topics. Provide only the specified output: "drug" or "end".
    """
    guardrails_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                guardrails_system,
            ),
            (
                "human",
                ("{question}"),
            ),
        ]
    )


    class GuardrailsOutput(BaseModel):
        decision: Literal["drug", "end"] = Field(
            description="Decision on whether the question is related to genes, drugs, and their targets"
        )


    guardrails_chain = guardrails_prompt | llm.with_structured_output(GuardrailsOutput)


    def guardrails(state: InputState) -> OverallState:
        """
        Decides if the question is related to genes, drugs, and their targets or not.
        """
        guardrails_output = guardrails_chain.invoke({"question": state.get("question")})
        database_records = None
        if guardrails_output.decision == "end":
            database_records = "This questions is not about genes, drugs, and their targets. Therefore I cannot answer this question."
        return {
            "next_action": guardrails_output.decision,
            "database_records": database_records,
            "steps": ["guardrail"],
        }


    with open('few_shot_prompts.json', 'r') as f:
        examples = json.load(f)

    #For future: change k number of examples to use?
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples, OpenAIEmbeddings(), Neo4jVector, k=5, input_keys=["question"]
    )

    text2cypher_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Given an input question, convert it to a Cypher query. No pre-amble."
                    "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
                ),
            ),
            (
                "human",
                (
                    """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
    Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
    Here is the schema information. It is important to know that there is also an 'embedding' property on Molecules that is very useful
    and part of the schema.
    {schema}

    Below are a number of examples of questions and their corresponding Cypher queries.
    IMPORTANT: When the question asks for similar drugs or similarity search, you MUST use this exact Cypher pattern with the embedding property
    on MOLECULE nodes:
    CALL db.index.vector.queryNodes('molEmbed', 10, ref.embedding)

    {fewshot_examples}

    User input: {question}
    Cypher query:"""
                ),
            ),
        ]
    )

    text2cypher_chain = text2cypher_prompt | llm | StrOutputParser()

    def generate_cypher(state: OverallState) -> OverallState:
        """
        Generates a cypher statement based on the provided schema and user input
        """
        NL = "\n"
        fewshot_examples = (NL * 2).join(
            [
                f"Question: {el['question']}{NL}Cypher:{el['query']}"
                for el in example_selector.select_examples(
                    {"question": state.get("question")}
                )
            ]
        )
        generated_cypher = text2cypher_chain.invoke(
            {
                "question": state.get("question"),
                "fewshot_examples": fewshot_examples,
                "schema": graph_with_schema.schema,
            }
        )
        return {"cypher_statement": generated_cypher, "steps": ["generate_cypher"]}

   
    validate_cypher_system = """
    You are a Cypher expert reviewing a statement written by a junior developer.
    """

    validate_cypher_user = """You must check the following:
    * Are there any syntax errors in the Cypher statement?
    * Are there any missing or undefined variables in the Cypher statement?
    * Are any node labels missing from the schema?
    * Are any relationship types missing from the schema?
    * Are any of the properties not included in the schema?
    * Does the Cypher statement include enough information to answer the question?

    Examples of good errors:
    * Label (:Foo) does not exist, did you mean (:Bar)?
    * Property bar does not exist for label Foo, did you mean baz?
    * Relationship FOO does not exist, did you mean FOO_BAR?

    Schema:
    {schema}

    The question is:
    {question}

    The Cypher statement is:
    {cypher}

    Make sure you don't make any mistakes!"""

    validate_cypher_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                validate_cypher_system,
            ),
            (
                "human",
                (validate_cypher_user),
            ),
        ]
    )


    class Property(BaseModel):
        """
        Represents a filter condition based on a specific node property in a graph in a Cypher statement.
        """

        node_label: str = Field(
            description="The label of the node to which this property belongs."
        )
        property_key: str = Field(description="The key of the property being filtered.")
        property_value: str = Field(
            description="The value that the property is being matched against."
        )


    class ValidateCypherOutput(BaseModel):
        """
        Represents the validation result of a Cypher query's output,
        including any errors and applied filters.
        """

        errors: Optional[List[str]] = Field(
            description="A list of syntax or semantical errors in the Cypher statement. Always explain the discrepancy between schema and Cypher statement"
        )
        filters: Optional[List[Property]] = Field(
            description="A list of property-based filters applied in the Cypher statement."
        )


    validate_cypher_chain = validate_cypher_prompt | llm.with_structured_output(
        ValidateCypherOutput
    )

    corrector_schema = [
        Schema(el["start"], el["type"], el["end"])
        for el in graph_with_schema.structured_schema.get("relationships")
    ]
    cypher_query_corrector = CypherQueryCorrector(corrector_schema)

    def validate_cypher(state: OverallState) -> OverallState:
        """
        Validates the Cypher statements and maps any property values to the database.
        """
        errors = []
        mapping_errors = []
        # Check for syntax errors
        try:
            graph_with_schema.query(f"EXPLAIN {state.get('cypher_statement')}")
        except CypherSyntaxError as e:
            errors.append(e.message)
        # Experimental feature for correcting relationship directions
        corrected_cypher = cypher_query_corrector(state.get("cypher_statement"))
        if not corrected_cypher:
            errors.append("The generated Cypher statement doesn't fit the graph schema")
        if not corrected_cypher == state.get("cypher_statement"):
            print("Relationship direction was corrected")
        # Use LLM to find additional potential errors and get the mapping for values
        llm_output = validate_cypher_chain.invoke(
            {
                "question": state.get("question"),
                "schema": graph_with_schema.schema,
                "cypher": state.get("cypher_statement"),
            }
        )
        if llm_output.errors:
            errors.extend(llm_output.errors)
        if llm_output.filters:
            for filter in llm_output.filters:
                # Do mapping only for string values
                if (
                    not [
                        prop
                        for prop in graph_with_schema.structured_schema["node_props"][
                            filter.node_label
                        ]
                        if prop["property"] == filter.property_key
                    ][0]["type"]
                    == "STRING"
                ):
                    continue
                mapping = graph_with_schema.query(
                    f"MATCH (n:{filter.node_label}) WHERE toLower(n.`{filter.property_key}`) = toLower($value) RETURN 'yes' LIMIT 1",
                    {"value": filter.property_value},
                )
                if not mapping:
                    print(
                        f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                    )
                    mapping_errors.append(
                        f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                    )
        if mapping_errors:
            next_action = "end"
        elif errors:
            next_action = "correct_cypher"
        else:
            next_action = "execute_cypher"

        return {
            "next_action": next_action,
            "cypher_statement": corrected_cypher,
            "cypher_errors": errors,
            "steps": ["validate_cypher"],
        }

    correct_cypher_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a Cypher expert reviewing a statement written by a junior developer. "
                    "You need to correct the Cypher statement based on the provided errors. No pre-amble."
                    "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
                ),
            ),
            (
                "human",
                (
                    """Check for invalid syntax or semantics and return a corrected Cypher statement.

    Schema:
    {schema}

    Note: Do not include any explanations or apologies in your responses.
    Do not wrap the response in any backticks or anything else.
    Respond with a Cypher statement only!

    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

    The question is:
    {question}

    The Cypher statement is:
    {cypher}

    The errors are:
    {errors}

    Corrected Cypher statement: """
                ),
            ),
        ]
    )

    correct_cypher_chain = correct_cypher_prompt | llm | StrOutputParser()


    def correct_cypher(state: OverallState) -> OverallState:
        """
        Correct the Cypher statement based on the provided errors.
        """
        corrected_cypher = correct_cypher_chain.invoke(
            {
                "question": state.get("question"),
                "errors": state.get("cypher_errors"),
                "cypher": state.get("cypher_statement"),
                "schema": graph_with_schema.schema,
            }
        )

        return {
            "next_action": "validate_cypher",
            "cypher_statement": corrected_cypher,
            "steps": ["correct_cypher"],
        }


    no_results = "I couldn't find any relevant information in the database"


    def execute_cypher(state: OverallState) -> OverallState:
        """
        Executes the given Cypher statement.
        """
        #print("Cypher to run:", state.get("cypher_statement"))
        records = graph_with_schema.query(state.get("cypher_statement"))
        return {
            "database_records": records if records else no_results,
            "next_action": "end",
            "steps": ["execute_cypher"],
        }

    """The final step is to generate the answer. This involves combining the initial question with the database output to produce a relevant response."""

    generate_final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant",
            ),
            (
                "human",
                (
                    """Use the following results retrieved from a database to provide
    a succinct, definitive answer to the user's question.

    Respond as if you are answering the question directly.

    Results: {results}
    Question: {question}"""
                ),
            ),
        ]
    )

    generate_final_chain = generate_final_prompt | llm | StrOutputParser()


    def generate_final_answer(state: OverallState) -> OutputState:
        """
        Decides if the question is related to genes, drugs, or their targets.
        """
        final_answer = generate_final_chain.invoke(
            {"question": state.get("question"), "results": state.get("database_records")}
        )
        return {"answer": final_answer, "steps": ["generate_final_answer"]}

    def guardrails_condition(
        state: OverallState,
    ) -> Literal["generate_cypher", "reject_question"]:
        if state.get("next_action") == "end":
            return "reject_question"
        elif state.get("next_action") == "drug":
            return "generate_cypher"


    def validate_cypher_condition(
        state: OverallState,
    ) -> Literal["generate_final_answer", "correct_cypher", "execute_cypher"]:
        if state.get("next_action") == "end":
            return "generate_final_answer"
        elif state.get("next_action") == "correct_cypher":
            return "correct_cypher"
        elif state.get("next_action") == "execute_cypher":
            return "execute_cypher"


    langgraph = StateGraph(OverallState, input=InputState, output=OutputState)
    langgraph.add_node(guardrails)
    langgraph.add_node(generate_cypher)
    langgraph.add_node("reject_question", reject_question)
    langgraph.add_node(validate_cypher)
    langgraph.add_node(correct_cypher)
    langgraph.add_node(execute_cypher)
    langgraph.add_node(generate_final_answer)

    langgraph.add_edge(START, "guardrails")
    langgraph.add_conditional_edges(
        "guardrails",
        guardrails_condition,
    )
    langgraph.add_edge("generate_cypher", "validate_cypher")
    langgraph.add_conditional_edges(
        "validate_cypher",
        validate_cypher_condition,
    )
    langgraph.add_edge("execute_cypher", "generate_final_answer")
    langgraph.add_edge("correct_cypher", "validate_cypher")
    langgraph.add_edge("generate_final_answer", END)
    langgraph.add_edge("reject_question", END)

    langgraph = langgraph.compile()

    return langgraph
