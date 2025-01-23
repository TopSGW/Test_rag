import os
import shutil
import warnings

import kuzu
import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.kuzu import KuzuPropertyGraphStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.llms.ollama import Ollama
from entity_extractor import EntityRelationshipExtractor

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

# Initialize the graph store without schema validation
graph_store = KuzuPropertyGraphStore(
    db,
    has_structured_schema=False,  # Disable schema validation
)

# --- Step 3: Use EntityRelationshipExtractor to automatically extract entities and relationships ---

# Initialize the extractor
extractor = EntityRelationshipExtractor(
    llm_model="llama3.3:70b",
    base_url="http://localhost:11434",
    cache_enabled=True
)

# Process all documents to extract entities and relationships
texts = [doc.text for doc in original_documents]
entities, relationships = extractor.process_documents(texts, show_progress=True)

# Open a connection to the database to modify the graph
conn = kuzu.Connection(db)

# Track created entity types to avoid duplicate table creation
created_entity_types = set()

# Create tables and add entities dynamically
for entity in entities:
    entity_type = entity.type.upper()
    
    # Create table if it doesn't exist
    if entity_type not in created_entity_types:
        conn.execute(f"CREATE NODE TABLE IF NOT EXISTS {entity_type} (id STRING, name STRING, PRIMARY KEY (id))")
        created_entity_types.add(entity_type)
    
    # Add entity
    conn.execute(
        f"""
        MERGE (e:{entity_type} {{id: $name, name: $name}})
        """,
        parameters={"name": entity.name},
    )

# Add relationships
for rel in relationships:
    source_type = rel.source.type.upper()
    target_type = rel.target.type.upper()
    relation_type = rel.relation_type.upper()
    
    # Create relationship if it doesn't exist
    conn.execute(
        f"""
        MATCH (s:{source_type} {{id: $source_name}})
        MATCH (t:{target_type} {{id: $target_name}})
        MERGE (s)-[r:{relation_type}]->(t)
        """,
        parameters={
            "source_name": rel.source.name,
            "target_name": rel.target.name
        },
    )

# Add birth dates for known individuals (this information might not be extractable from text)
try:
    conn.execute("ALTER TABLE PERSON ADD birth_date STRING")
except RuntimeError:
    pass

birth_dates = {
    "Larry Fink": "1952-11-02",
    "Susan Wagner": "1961-05-26",
    "Robert Kapito": "1957-02-08"
}

for name, date in birth_dates.items():
    conn.execute(
        """
        MERGE (p:PERSON {id: $name})
        ON MATCH SET p.birth_date = $date
        """,
        parameters={"name": name, "date": date},
    )

conn.close()