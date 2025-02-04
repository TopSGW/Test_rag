import os

import kuzu
from dotenv import load_dotenv
import ell
from openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
import numpy as np

import prompts

load_dotenv()
SEED = 42

# Configure OpenAI client for Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't need a real API key
)

# Register the model with Ellama
MODEL = "llama3.3:70b"
ell.config.register_model(MODEL, client)

# Set up the embedding model
embed_model = OllamaEmbedding(
    model_name="llama3.3:70b",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

def get_relationship_embedding(rel_type: str) -> np.ndarray:
    """Generate embedding for a relationship type."""
    rel_text = f"This represents a {rel_type} relationship between entities"
    embedding = embed_model.get_text_embedding(rel_text)
    return np.array(embedding)

class GraphRAG:
    """Graph Retrieval Augmented Generation from a Kùzu database."""

    def __init__(self, db_path="./test_kuzudb"):
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)

    def get_schema(self) -> str:
        """Provides the graph schema information for the purposes of Cypher generation via an LLM."""
        try:
            # Get all node tables and their properties
            node_properties = []
            node_table_names = self.conn._get_node_table_names()
            for table_name in node_table_names:
                current_table_schema = {"properties": [], "label": table_name}
                properties = self.conn._get_node_property_names(table_name)
                for property_name in properties:
                    property_type = properties[property_name]["type"]
                    list_type_flag = ""
                    if properties[property_name]["dimension"] > 0:
                        if "shape" in properties[property_name]:
                            for s in properties[property_name]["shape"]:
                                list_type_flag += "[%s]" % s
                        else:
                            for i in range(properties[property_name]["dimension"]):
                                list_type_flag += "[]"
                    property_type += list_type_flag
                    current_table_schema["properties"].append((property_name, property_type))
                node_properties.append(current_table_schema)

            # Get all relationship tables and their structure
            relationships = []
            rel_tables = self.conn._get_rel_table_names()
            for table in rel_tables:
                # Format the relationship pattern using source_name and target_name
                relationships.append("(:%s)-[:%s {embedding: VAR_LIST<FLOAT>, description: STRING}]->(:%s)" % (table["src"], table["name"], table["dst"]))
                print(f"source:{table['src']} type::{table['name']}, target::{table['dst']}")

            # Get relationship properties
            rel_properties = []
            for table in rel_tables:
                table_name = table["name"]
                current_table_schema = {"properties": [], "label": table_name}
                query_result = self.conn.execute(f"CALL table_info('{table_name}') RETURN *;")
                while query_result.has_next():
                    row = query_result.get_next()
                    prop_name = row[1]
                    prop_type = row[2]
                    current_table_schema["properties"].append((prop_name, prop_type))
                rel_properties.append(current_table_schema)

            # Format the schema information for the LLM
            schema_parts = []
            
            # Add node information
            if node_properties:
                schema_parts.append("Node types and their properties:")
                for node in node_properties:
                    props = [f"{prop[0]} ({prop[1]})" for prop in node["properties"]]
                    schema_parts.append(f"  {node['label']}: {', '.join(props)}")

            # Add relationship patterns with vector similarity support
            if relationships:
                schema_parts.append("\nValid relationship patterns:")
                schema_parts.append("Note: All relationships support vector similarity matching using dot product.")
                schema_parts.append("Relationships store embeddings as VAR_LIST<FLOAT> for semantic similarity search.")
                for rel in relationships:
                    schema_parts.append(f"  {rel}")

            # Add relationship properties
            if rel_properties:
                schema_parts.append("\nRelationship properties:")
                for rel in rel_properties:
                    props = [f"{prop[0]} ({prop[1]})" for prop in rel["properties"]]
                    schema_parts.append(f"  {rel['label']}: {', '.join(props)}")

            # Combine all parts with proper formatting
            schema = "\n".join(schema_parts)
            
            # Add usage hints for the LLM
            schema += "\n\nNotes for querying:"
            schema += "\n- Use MATCH to find patterns in the graph"
            schema += "\n- Properties can be accessed using dot notation (node.property)"
            schema += "\n- Relationships are directional, use -> for direction"
            schema += "\n- For finding semantically similar relationships, use vector dot product:"
            schema += "\n  Example: MATCH (a)-[r]->(b)"
            schema += "\n          WITH r.description AS desc,"
            schema += "\n               r.embedding AS emb,"
            schema += "\n               $rel_embedding AS test_emb"
            schema += "\n          WHERE emb IS NOT NULL"
            schema += "\n          WITH desc,"
            schema += "\n               reduce(dot = 0.0, i IN RANGE(0, size(emb)-1) |"
            schema += "\n                  dot + emb[i] * test_emb[i]) /"
            schema += "\n               (sqrt(reduce(norm1 = 0.0, i IN RANGE(0, size(emb)-1) |"
            schema += "\n                  norm1 + emb[i] * emb[i])) *"
            schema += "\n                sqrt(reduce(norm2 = 0.0, i IN RANGE(0, size(test_emb)-1) |"
            schema += "\n                  norm2 + test_emb[i] * test_emb[i]))) AS similarity"
            schema += "\n          WHERE similarity > $threshold"
            schema += "\n          RETURN desc, similarity"
            schema += "\n          ORDER BY similarity DESC"
            
            return schema

        except Exception as e:
            # Return a simplified schema in case of errors
            return f"Error getting detailed schema. Basic query patterns are supported using MATCH clause with node labels and relationship types shown above. Error: {str(e)}"

    def query(self, question: str, cypher: str) -> str:
        """Use the generated Cypher statement to query the graph database."""
        try:
            # Generate embedding for the relationship implied by the question
            rel_embedding = get_relationship_embedding(question).tolist()
            
            # Execute query with relationship embedding for semantic matching
            response = self.conn.execute(
                cypher,
                parameters={
                    "rel_embedding": rel_embedding,
                    "threshold": 0.85  # Configurable similarity threshold
                }
            )

            result = []
            while response.has_next():
                item = response.get_next()
                if isinstance(item, (list, tuple)):
                    result.extend(item)
                else:
                    result.append(item)

            # Handle both hashable and non-hashable types
            if all(isinstance(x, (str, int, float, bool, tuple)) for x in result):
                final_result = {question: list(set(result))}
            else:
                # For non-hashable types, we'll use a list comprehension to remove duplicates
                final_result = {question: [x for i, x in enumerate(result) if x not in result[:i]]}

            return final_result
        except Exception as e:
            return {question: f"Error executing query: {str(e)}"}

    @ell.simple(model=MODEL, temperature=0.1)
    def generate_cypher(self, question: str) -> str:
        return [
            ell.system(prompts.CYPHER_SYSTEM_PROMPT),
            ell.user(
                prompts.CYPHER_USER_PROMPT.format(schema=self.get_schema(), question=question)
            ),
        ]

    @ell.simple(model=MODEL, temperature=0.3)
    def retrieve(self, question: str, context: str) -> str:
        return [
            ell.system(prompts.RAG_SYSTEM_PROMPT),
            ell.user(prompts.RAG_USER_PROMPT.format(question=question, context=context)),
        ]

    def run(self, question: str) -> str:
        cypher = self.generate_cypher(question)
        print(f"\n{cypher}\n")
        context = self.query(question, cypher)
        return self.retrieve(question, context)


if __name__ == "__main__":
    graph_rag = GraphRAG("./test_kuzudb")
    
    # Test queries with semantic relationship matching
    questions = [
        "Who are the founders of BlackRock? Return the names as a numbered list.",
        "Where did Larry Fink graduate from?",
        "When was Susan Wagner born?",
        "How did Larry Fink and Rob Kapito meet?",
        "What companies did Larry Fink establish?",
        "Who started BlackRock with Larry Fink?",
    ]
    
    for i, question in enumerate(questions, 1):
        response = graph_rag.run(question)
        print(f"\nQ{i}: {question}\n{response}\n---")