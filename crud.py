import os
import shutil
import warnings
from typing import Literal

import kuzu
import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.kuzu import KuzuPropertyGraphStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.llms.ollama import Ollama

# Load environment variables
load_dotenv()
SEED = 42
nest_asyncio.apply()

# Set up the embedding model using Ollama
embed_model = OllamaEmbedding(
    model_name="llama3.3:70b",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

# Configure Ollama LLMs
extraction_llm = Ollama(
    model="llama3.3:70b",
    temperature=0.0,
    request_timeout=120.0,
    base_url="http://localhost:11434"
)

generation_llm = Ollama(
    model="llama3.3:70b",
    temperature=0.3,
    request_timeout=120.0,
    base_url="http://localhost:11434"
)

# Load the dataset on Larry Fink
original_documents = SimpleDirectoryReader("./data/blackrock").load_data()

# --- Step 1: Chunk and store the vector embeddings in LanceDB ---
shutil.rmtree("./test_lancedb", ignore_errors=True)

vector_store = LanceDBVectorStore(
    uri="./test_lancedb",
    mode="overwrite",
)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=32),
        embed_model,
    ],
    vector_store=vector_store,
)
pipeline.run(documents=original_documents)

# Create the vector index
vector_index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
    llm=generation_llm,
)

# --- Step 2: Construct the graph in KùzuDB ---

shutil.rmtree("test_kuzudb", ignore_errors=True)
db = kuzu.Database("test_kuzudb")

warnings.filterwarnings("ignore")

# Define the allowed entities and relationships
entities = Literal["PERSON", "CITY", "STATE", "UNIVERSITY", "ORGANIZATION"]
relations = Literal[
    "STUDIED_AT",
    "IS_FOUNDER_OF",
    "IS_CEO_OF",
    "BORN_IN",
    "IS_CITY_IN",
]

validation_schema = [
    ("PERSON", "STUDIED_AT", "UNIVERSITY"),
    ("PERSON", "IS_CEO_OF", "ORGANIZATION"),
    ("PERSON", "IS_FOUNDER_OF", "ORGANIZATION"),
    ("PERSON", "BORN_IN", "CITY"),
    ("CITY", "IS_CITY_IN", "STATE"),
]

graph_store = KuzuPropertyGraphStore(
    db,
    has_structured_schema=True,
    relationship_schema=validation_schema,
)

schema_path_extractor = SchemaLLMPathExtractor(
    llm=extraction_llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    strict=True,
)

kg_index = PropertyGraphIndex.from_documents(
    original_documents,
    embed_model=embed_model,
    kg_extractors=[schema_path_extractor],
    property_graph_store=graph_store,
    show_progress=True,
)

# Step 3: Augment the graph with external knowledge and fix erroneous relationships

# Say we have this knowledge obtained from other sources about additional founders of BlackRock
additional_founders = [
    "Ben Golub",
    "Barbara Novick",
    "Ralph Schlosstein",
    "Keith Anderson",
    "Hugh Frater",
]

# Open a connection to the database to modify the graph
conn = kuzu.Connection(db)

# Add additional founder nodes of type PERSON to the graph store
for founder in additional_founders:
    conn.execute(
        """
        MATCH (o:ORGANIZATION {id: "BlackRock"})
        MERGE (p:PERSON {id: $name, name: $name})
        MERGE (p)-[r:IS_FOUNDER_OF]->(o)
        """,
        parameters={"name": founder},
    )

# Alter PERSON schema and add a birth_date property
try:
    conn.execute("ALTER TABLE PERSON ADD birth_date STRING")
except RuntimeError:
    pass

names = ["Larry Fink", "Susan Wagner", "Robert Kapito"]
dates = ["1952-11-02", "1961-05-26", "1957-02-08"]

for name, date in zip(names, dates):
    conn.execute(
        """
    MERGE (p:PERSON {id: $name})
    ON MATCH SET p.birth_date = $date
    """,
        parameters={"name": name, "date": date},
    )

conn.close()