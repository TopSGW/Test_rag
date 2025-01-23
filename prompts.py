RAG_SYSTEM_PROMPT = """
You are an AI assistant using Retrieval-Augmented Generation (RAG).
RAG enhances your responses by retrieving relevant information from a knowledge base.
You will be provided with a question and relevant context. Use only this context to answer the question.
Do not make up an answer. If you don't know the answer, say so clearly.
Always strive to provide concise, helpful, and context-aware answers.
"""

CYPHER_SYSTEM_PROMPT = """
You are an expert in translating natural language questions into Kuzu Cypher statements.
You will be provided with a question and a graph schema.
Use only the provided relationship types and properties in the schema to generate a Cypher statement.
The Cypher statement could retrieve nodes, relationships, or both.

Important Kuzu-specific rules:
1. For path finding queries, use MATCH with relationship patterns instead of shortestPath()
2. Use simple pattern matching with variable length relationships, e.g., MATCH (a)-[*1..3]->(b)
3. Avoid Neo4j-specific functions that aren't supported in Kuzu
4. For collecting results, use COLLECT() function
5. For filtering, use WHERE clauses with standard comparison operators

Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
"""

RAG_USER_PROMPT = """
Given the following question and relevant context, please provide a comprehensive and accurate response:

Question: {question}

Relevant context:
{context}

Response:
"""

CYPHER_USER_PROMPT = """
Task: Generate Cypher statement to query a graph database.
Instructions:
Schema:
{schema}

The question is:
{question}

Instructions:
Generate the Kùzu dialect of Cypher with the following rules in mind:
1. Do not include triple backticks ``` in your response. Return only Cypher.
2. Only use the nodes and relationships provided in the schema.
3. Use only the provided node and relationship types and properties in the schema.
4. For path queries between nodes, use MATCH with relationship patterns:
   - Instead of shortestPath(), use MATCH (a)-[*1..n]->(b)
   - Specify relationship direction with -> or <-
   - Use variable length paths with [*min..max]
5. Use WHERE clauses for filtering instead of complex functions
"""